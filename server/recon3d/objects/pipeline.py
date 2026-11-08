"""capture.zip + scene.splat/scene.ply -> objects.zip (object table), see spec 2026-11-07 §4.4."""
from __future__ import annotations

import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from recon3d.capture.poses import gl_to_cv_c2w
from recon3d.config import Profile
from recon3d.jobs.pipeline import OnProgress, _log
from recon3d.objects import assign as assign_mod
from recon3d.objects import merge as merge_mod
from recon3d.objects import table
from recon3d.objects.relations import compute_relations
from recon3d.objects.render_depth import read_splat_positions
from recon3d.objects.thumbs import crop_thumbnail
from recon3d.objects.vocab import zh_label

STAGE_DETECT = "识别物体 (检测)"
STAGE_MERGE = "识别物体 (合并)"
STAGE_ASSIGN = "识别物体 (归属)"
STAGE_THUMBS = "识别物体 (截图)"
MAX_FRAMES = 240
DEPTH_SCALE = 0.5
MAX_POINTS_PER_DETECTION = 2000
# Sampled frames are ~1 s apart on a dense trajectory: an object "seen" in only a handful of them is
# almost always a flicker (first office run: 87 of 169 objects in 2 frames, mostly windows/doors/lamps
# that were not there; at MIN_VIEWS=3 the phone review still found the 3-5 frame ones to be noise).
MIN_VIEWS = 6

# Injection points for tests (resolved lazily: the real ones import torch/transformers).
detector_factory = None  # (device) -> object with detect(PIL.Image) -> list[FrameDetection], close()
render_depths_fn = None  # same signature as render_depth.render_depths


class ModelDownloadFailed(Exception):
    """Detector weights could not be downloaded (network); user text set by the worker."""


def sample_indices(n: int) -> list[int]:
    k = max(1, math.ceil(n / MAX_FRAMES))
    return list(range(0, n, k))


def _detector(device: str):
    if detector_factory is not None:
        return detector_factory(device)
    from recon3d.objects.detect import Detector

    return Detector(device)


def _render_depths(*args, **kwargs):
    fn = render_depths_fn
    if fn is None:
        from recon3d.objects.render_depth import render_depths as fn
    return fn(*args, **kwargs)


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def _shrink(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(mask).resize((w, h), Image.NEAREST), dtype=bool)


def _backproject(mask_small: np.ndarray, depth: np.ndarray, K_small: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """World points of the mask pixels that have rendered depth, subsampled to MAX_POINTS_PER_DETECTION."""
    vs, us = np.nonzero(mask_small & (depth > 0))
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(vs) > MAX_POINTS_PER_DETECTION:
        pick = np.linspace(0, len(vs) - 1, MAX_POINTS_PER_DETECTION).astype(int)
        vs, us = vs[pick], us[pick]
    z = depth[vs, us].astype(np.float64)
    x = (us + 0.5 - K_small[0, 2]) / K_small[0, 0] * z
    y = (vs + 0.5 - K_small[1, 2]) / K_small[1, 1] * z
    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (cam @ c2w.T)[:, :3]


def run_objects(job_dir: Path, profile: Profile, on_progress: OnProgress) -> int:
    """Builds the object table for a finished job; returns the number of objects."""
    from recon3d.capture.format import load_capture
    from recon3d.objects.detect import DownloadError

    job_dir = Path(job_dir)
    workdir = job_dir / "capture"
    out_dir = job_dir / "objects"
    device = _device()
    detector = None
    t_start = time.monotonic()
    try:
        on_progress(0.0, STAGE_DETECT)
        cap = load_capture(job_dir / "capture.zip", workdir)
        idx = sample_indices(len(cap.frames))
        frames = [cap.frames[i] for i in idx]
        W, H = cap.manifest.image_width, cap.manifest.image_height
        poses_path = job_dir / "poses_cv.npy"
        if poses_path.is_file():
            c2w = np.asarray(np.load(poses_path)[idx], dtype=np.float64)
        else:
            c2w = np.stack([gl_to_cv_c2w(f.pose) for f in frames])
        Ks = np.array([[[f.fx, 0, f.cx], [0, f.fy, f.cy], [0, 0, 1]] for f in frames], dtype=np.float64)
        h, w = round(H * DEPTH_SCALE), round(W * DEPTH_SCALE)
        Ks_small = Ks.copy()
        Ks_small[:, 0, :] *= w / W
        Ks_small[:, 1, :] *= h / H

        # --- detect + segment (0 -> 0.6); depth rendered per frame for back-projection ---
        t0 = time.monotonic()
        try:
            detector = _detector(device)
        except DownloadError as e:
            raise ModelDownloadFailed(str(e)) from e
        dets: list[merge_mod.Detection] = []
        # Depth-resolution masks, bit-packed (a full-res SAM mask is ~2 MB; 1000+ detections would hold
        # gigabytes), plus each mask's full-res area for picking the thumbnail; both parallel to dets.
        det_masks: list[np.ndarray] = []
        det_areas: list[float] = []
        depths: list[np.ndarray] = []
        depth_iter = _render_depths(job_dir / "scene.ply", c2w, Ks, W, H, device=device, scale=DEPTH_SCALE)
        for fi, (frame, depth) in enumerate(zip(frames, depth_iter)):
            depths.append(depth)
            img = Image.open(cap.root / frame.image).convert("RGB")
            try:
                found = detector.detect(img)
            except DownloadError as e:
                raise ModelDownloadFailed(str(e)) from e
            for fd in found:
                small = _shrink(fd.mask, w, h)
                pts = _backproject(small, depth, Ks_small[fi], c2w[fi])
                dets.append(merge_mod.Detection(fi, fd.label, fd.score, fd.box, pts))
                det_masks.append(np.packbits(small))
                det_areas.append(float(fd.mask.sum()))
            del found
            on_progress(0.6 * (fi + 1) / len(frames), STAGE_DETECT)
        detector.close()
        detector = None
        _log(job_dir, f"objects: detect {len(frames)} frames, {len(dets)} detections, {time.monotonic() - t0:.1f}s")

        # --- merge (0.6 -> 0.75) ---
        t0 = time.monotonic()
        on_progress(0.6, STAGE_MERGE)
        merged = merge_mod.merge_detections(dets, min_views=MIN_VIEWS)
        _log(job_dir, f"objects: merge -> {len(merged)} objects, {time.monotonic() - t0:.1f}s")

        # --- assign (0.75 -> 0.95) ---
        t0 = time.monotonic()
        on_progress(0.75, STAGE_ASSIGN)
        positions = read_splat_positions(job_dir / "scene.splat")
        masks = [
            assign_mod.ObjectMasks(dets[di].frame, oi + 1, np.unpackbits(det_masks[di])[: h * w].reshape(h, w).astype(bool))
            for oi, obj in enumerate(merged)
            for di in obj.members
        ]
        if merged:
            ids = assign_mod.assign_gaussians(
                positions, c2w, Ks_small, depths, masks,
                bbox_min=np.stack([o.bbox_min for o in merged]),
                bbox_max=np.stack([o.bbox_max for o in merged]),
                device=device,
            )
        else:
            ids = np.zeros(len(positions), dtype=np.uint16)
        # An object without a single Gaussian cannot be highlighted: drop it and renumber the rest 1..N.
        counts = np.bincount(ids, minlength=len(merged) + 1)
        keep = [oi for oi in range(len(merged)) if counts[oi + 1] > 0]
        remap = np.zeros(len(merged) + 1, dtype=np.uint16)
        for new_id, oi in enumerate(keep, start=1):
            remap[oi + 1] = new_id
        ids = remap[ids]
        dropped = len(merged) - len(keep)
        merged = [merged[oi] for oi in keep]
        _log(
            job_dir,
            f"objects: assign {len(positions)} gaussians, {int((ids > 0).sum())} assigned, "
            f"{dropped} objects without gaussians dropped, {time.monotonic() - t0:.1f}s",
        )

        # --- thumbnails + relations + table (0.95 -> 1.0) ---
        t0 = time.monotonic()
        on_progress(0.95, STAGE_THUMBS)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "thumbs").mkdir(parents=True)
        records: list[table.ObjectRecord] = []
        for oi, obj in enumerate(merged):
            oid = oi + 1
            best = max(obj.members, key=lambda di: dets[di].score * det_areas[di])
            img = Image.open(cap.root / frames[dets[best].frame].image).convert("RGB")
            crop_thumbnail(img, dets[best].box).save(out_dir / "thumbs" / f"{oid}.jpg", quality=85)
            records.append(
                table.ObjectRecord(
                    id=oid,
                    label=obj.label,
                    label_zh=zh_label(obj.label),
                    score=float(obj.score),
                    views=int(obj.views),
                    center=tuple(float(v) for v in obj.center),
                    bbox_min=tuple(float(v) for v in obj.bbox_min),
                    bbox_max=tuple(float(v) for v in obj.bbox_max),
                    gaussian_count=int((ids == oid).sum()),
                )
            )
        home_look = c2w[0][:3, 2] if len(c2w) else None  # OpenCV +Z = view direction
        relations = compute_relations(records, home_look)
        table.write_objects(out_dir, job_dir.name, len(frames), ids, records, relations)
        _log(job_dir, f"objects: thumbs+relations {time.monotonic() - t0:.1f}s; total {time.monotonic() - t_start:.1f}s")
        on_progress(1.0, None)
        return len(records)
    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception:  # pragma: no cover
                traceback.print_exc()
        shutil.rmtree(workdir, ignore_errors=True)
