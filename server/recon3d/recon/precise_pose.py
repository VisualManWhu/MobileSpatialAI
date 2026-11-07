"""Precise mode: refine ARCore poses with pycolmap, keeping the ARCore world frame.

Pipeline (CPU): SIFT -> sequential + spatial matching (ARCore positions as the
spatial index) -> known-pose triangulation from the ARCore poses -> COLMAP
mapper global bundle adjustment with position priors (intrinsics fixed). The
priors keep the refined poses in the ARCore metric frame, so the result feeds
straight into the existing depth-initialised training.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recon3d.capture.format import Capture
from recon3d.capture.poses import gl_to_cv_c2w

log = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 1600
MAX_FEATURES = 4096
SEQ_OVERLAP = 10
SPATIAL_NEIGHBORS = 20
SPATIAL_MAX_DIST_M = 3.0
PRIOR_SIGMA_M = 0.3
BA_GLOBAL_REFINEMENTS = 3
MIN_POINTS_REGISTERED = 20
MIN_REGISTERED_RATIO = 0.7

# Progress split inside precise_pose's own 0..1.
_P_EXTRACT_END = 0.35
_P_SEQ_END = 0.55
_P_MATCH_END = 0.7
_P_MAPPER_CAP = 0.95
_MAPPER_POLL_S = 2.0


@dataclass
class PrecisePoseResult:
    c2w_cv: np.ndarray      # [N,4,4] float64, OpenCV convention, ARCore world frame
    registered: np.ndarray  # [N] bool
    stats: dict


class PrecisePoseFailed(Exception):
    """Too few frames registered (or the mapper produced nothing usable)."""

    def __init__(self, ratio: float, cause: BaseException | None = None):
        self.ratio = float(ratio)
        self.cause = cause
        msg = f"precise pose failed: {self.ratio:.0%} of frames registered (need {MIN_REGISTERED_RATIO:.0%})"
        if cause is not None:
            msg += f"; cause: {cause!r}"
        super().__init__(msg)


def _iid(x):
    return x.id if hasattr(x, "id") else x


def _rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) with dst ~= s * R @ src + t, full least squares."""
    ms, md = src.mean(0), dst.mean(0)
    sc, dc = src - ms, dst - md
    U, S, Vt = np.linalg.svd(dc.T @ sc / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    var_src = (sc ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_src) if var_src > 0 else 1.0
    t = md - s * R @ ms
    return s, R, t


def precise_pose(cap: Capture, workdir: Path, on_progress: Callable[[float], None]) -> PrecisePoseResult:
    import pycolmap

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    image_dir = cap.root / "images"
    n = len(cap.frames)
    names: list[str] = []
    for fr in cap.frames:
        assert fr.image.startswith("images/"), fr.image
        names.append(fr.image[len("images/"):])
    c2w_arcore = np.stack([gl_to_cv_c2w(fr.pose) for fr in cap.frames])
    idx_by_name = {nm: i for i, nm in enumerate(names)}

    # One COLMAP camera per rounded intrinsic set.
    groups: dict[tuple[float, float, float, float], list[str]] = {}
    for fr, nm in zip(cap.frames, names):
        groups.setdefault(tuple(round(v, 1) for v in (fr.fx, fr.fy, fr.cx, fr.cy)), []).append(nm)

    db = workdir / "db.db"
    if db.exists():
        db.unlink()
    threads = os.cpu_count() or 4

    t0 = time.time()
    eo = pycolmap.FeatureExtractionOptions()
    eo.max_image_size = MAX_IMAGE_SIZE
    eo.sift.max_num_features = MAX_FEATURES
    eo.num_threads = threads
    for gi, ((fx, fy, cx, cy), group_names) in enumerate(groups.items()):
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "PINHOLE"
        ro.camera_params = f"{fx},{fy},{cx},{cy}"
        pycolmap.extract_features(
            db, image_dir, image_names=group_names, camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=ro, extraction_options=eo,
        )
        on_progress(_P_EXTRACT_END * (gi + 1) / len(groups))
    t_extract = time.time() - t0

    database = pycolmap.Database.open(db)
    imgs = database.read_all_images()
    img_by_id = {im.image_id: im for im in imgs}
    database.clear_pose_priors()
    cov = np.diag([PRIOR_SIGMA_M ** 2] * 3)
    for f in database.read_all_frames():
        im = img_by_id[_iid(list(f.image_ids)[0])]
        prior = pycolmap.PosePrior(
            position=c2w_arcore[idx_by_name[im.name]][:3, 3],
            coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
            corr_data_id=list(f.data_ids)[0],
            position_covariance=cov,
        )
        database.write_pose_prior(prior)
    database.close()

    t0 = time.time()
    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads = threads
    sp = pycolmap.SequentialPairingOptions()
    sp.overlap = SEQ_OVERLAP
    sp.loop_detection = False
    pycolmap.match_sequential(db, matching_options=mo, pairing_options=sp)
    on_progress(_P_SEQ_END)
    so = pycolmap.SpatialPairingOptions()
    so.max_num_neighbors = SPATIAL_NEIGHBORS
    so.max_distance = SPATIAL_MAX_DIST_M
    so.ignore_z = False
    pycolmap.match_spatial(db, matching_options=mo, pairing_options=so)
    on_progress(_P_MATCH_END)
    t_match = time.time() - t0

    database = pycolmap.Database.open(db)
    imgs = database.read_all_images()
    img_by_id = {im.image_id: im for im in imgs}
    frames_db = database.read_all_frames()
    cams = database.read_all_cameras()
    rigs = database.read_all_rigs()
    database.close()

    rec = pycolmap.Reconstruction()
    for c in cams:
        rec.add_camera(c)
    for r in rigs:
        rec.add_rig(r)
    for f in frames_db:
        fr = pycolmap.Frame()
        fr.frame_id = f.frame_id
        fr.rig_id = f.rig_id
        for d in f.data_ids:
            fr.add_data_id(d)
        im = img_by_id[_iid(list(f.image_ids)[0])]
        w2c = np.linalg.inv(c2w_arcore[idx_by_name[im.name]])
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3])
        rec.add_frame(fr)
    for im in imgs:
        rec.add_image(pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=im.image_id, frame_id=im.frame_id))
    for f in frames_db:
        rec.register_frame(f.frame_id)

    tri_dir, init_dir, out_dir = workdir / "tri", workdir / "init", workdir / "out"
    for d in (tri_dir, init_dir, out_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    rec = pycolmap.triangulate_points(rec, db, image_dir, tri_dir, refine_intrinsics=False)
    rec.write(init_dir)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.ba_global_max_refinements = BA_GLOBAL_REFINEMENTS
    opts.use_prior_position = True
    opts.use_robust_loss_on_prior_position = True
    opts.prior_position_loss_scale = PRIOR_SIGMA_M
    opts.num_threads = threads

    t0 = time.time()
    result: dict = {}

    def _run_mapper() -> None:
        try:
            result["recs"] = pycolmap.incremental_mapping(db, image_dir, out_dir, options=opts, input_path=init_dir)
        except BaseException as e:  # surfaced in the caller
            result["error"] = e

    th = threading.Thread(target=_run_mapper, name="colmap-mapper", daemon=True)
    th.start()
    estimate = max(1.5 * (t_extract + t_match), 1.0)
    while th.is_alive():
        th.join(_MAPPER_POLL_S)
        frac = min(1.0, (time.time() - t0) / estimate)
        on_progress(_P_MATCH_END + (_P_MAPPER_CAP - _P_MATCH_END) * frac)
    t_mapper = time.time() - t0
    if "error" in result:
        raise PrecisePoseFailed(0.0, result["error"])
    recs = result.get("recs") or {}
    if not recs:
        raise PrecisePoseFailed(0.0)
    best = max(recs.values(), key=lambda r: r.num_reg_images())

    # Registered = has a pose and enough triangulated observations.
    registered = np.zeros(n, dtype=bool)
    c2w_new = c2w_arcore.copy()
    for im in imgs:
        i = idx_by_name[im.name]
        if im.image_id not in best.images:
            continue
        I = best.image(im.image_id)
        if not I.has_pose:
            continue
        if sum(1 for p in I.points2D if p.has_point3D()) < MIN_POINTS_REGISTERED:
            continue
        cfw = np.asarray(I.cam_from_world().matrix())
        c2w_new[i] = np.linalg.inv(np.vstack([cfw[:3], [0.0, 0.0, 0.0, 1.0]]))
        registered[i] = True

    ratio = float(registered.mean()) if n else 0.0
    if ratio < MIN_REGISTERED_RATIO:
        raise PrecisePoseFailed(ratio)

    # Unregistered frames inherit the nearest registered frame's rigid correction.
    reg_idx = np.nonzero(registered)[0]
    for i in np.nonzero(~registered)[0]:
        j = reg_idx[np.argmin(np.abs(reg_idx - i))]  # argmin -> first (smaller j) on ties
        c2w_new[i] = c2w_new[j] @ np.linalg.inv(c2w_arcore[j]) @ c2w_arcore[i]

    # Umeyama sanity check (BA -> ARCore), statistic only: priors already pin the frame.
    scale, R_al, _ = _umeyama(c2w_new[registered][:, :3, 3], c2w_arcore[registered][:, :3, 3])
    rot_al = _rotation_angle_deg(R_al)
    if abs(scale - 1.0) > 0.05 or rot_al > 2.0:
        log.warning("precise pose: BA frame drifted from ARCore (scale %.3f, rot %.2f deg)", scale, rot_al)

    shift = np.linalg.norm(c2w_new[:, :3, 3] - c2w_arcore[:, :3, 3], axis=1)
    rot = np.array([_rotation_angle_deg(c2w_new[i, :3, :3] @ c2w_arcore[i, :3, :3].T) for i in range(n)])
    stats = dict(
        frames=n, registered=int(registered.sum()), registered_ratio=ratio, cameras=len(cams),
        points3d=int(best.num_points3D()), reproj_px=float(best.compute_mean_reprojection_error()),
        shift_mean_m=float(shift.mean()), shift_median_m=float(np.median(shift)), shift_max_m=float(shift.max()),
        rot_mean_deg=float(rot.mean()), scale=scale, rot_deg=rot_al,
        t_extract_s=t_extract, t_match_s=t_match, t_mapper_s=t_mapper,
    )
    log.info("precise pose: %s", " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
    on_progress(1.0)
    return PrecisePoseResult(c2w_cv=c2w_new, registered=registered, stats=stats)
