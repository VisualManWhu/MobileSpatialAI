"""capture.zip + scene.ply/scene.splat -> objects.zip, v2 (spec 2026-11-08 §4):
structure (planes) -> tracking (identity) -> verification (size/support + VLM) -> voxel assignment."""
from __future__ import annotations

import random
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
from recon3d.objects.relations import compute_relations
from recon3d.objects.render_depth import read_splat_positions
from recon3d.objects.segments import Segment, merge_segments, support_of, track_to_segment
from recon3d.objects.structure import Room, Wall, depth_point_cloud, fit_room
from recon3d.objects.table import ObjectRecord, write_objects
from recon3d.objects.thumbs import crop_thumbnail
from recon3d.objects.verify import Verified, candidate_entries, confidence_of, confirm, door_window_ok, harmony_filter

STAGE_STRUCTURE = "识别物体 (结构)"
STAGE_TRACK = "识别物体 (跟踪)"
STAGE_VERIFY = "识别物体 (确认)"
STAGE_ASSIGN = "识别物体 (归属)"
DEPTH_SCALE = 0.5
CROP_PAD = 0.2
STRUCTURE_ZH = {"floor": "地面", "ceiling": "天花板", "wall": "墙面"}

# Injection points for tests (the real ones import torch / transformers lazily).
tracker_factory = None  # (device, profile) -> object with run(paths, mask_hw, on_frame) -> list[Track], close()
vlm_factory = None  # (device, profile) -> object with ask(images, prompt) -> str, close()
render_depths_fn = None  # same signature as render_depth.render_depths


class ModelDownloadFailed(Exception):
    """Detector / tracker / VLM weights could not be downloaded (network); user text set by the worker."""


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def _tracker(device: str, profile: Profile):
    if tracker_factory is not None:
        return tracker_factory(device, profile)
    from recon3d.objects.track import Tracker

    return Tracker(device, profile.objects_detector, profile.objects_sam)


def _vlm(device: str, profile: Profile):
    if vlm_factory is not None:
        return vlm_factory(device, profile)
    from recon3d.objects.vlm import QwenVlm

    return QwenVlm(profile.objects_vlm, device, four_bit=profile.objects_vlm_4bit)


def _render_depths(*args, **kwargs):
    fn = render_depths_fn
    if fn is None:
        from recon3d.objects.render_depth import render_depths as fn
    return fn(*args, **kwargs)


def _free_gb(device: str) -> float:
    if device != "cuda":
        return 0.0
    try:
        import torch

        free, _ = torch.cuda.mem_get_info()
        return free / 2**30
    except Exception:  # pragma: no cover
        return 0.0


def _crop(img: Image.Image, box, pad: float = CROP_PAD) -> Image.Image:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0, x0 - pad * w), max(0, y0 - pad * h)
    x1, y1 = min(img.width, x1 + pad * w), min(img.height, y1 + pad * h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return img.copy()
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def _crops(seg: Segment, frames, root: Path) -> list[Image.Image]:
    out = []
    for fi, box in seg.best_views[:5]:
        img = Image.open(root / frames[fi].image).convert("RGB")
        out.append(_crop(img, box))
    return out


def _wall_bbox(wall: Wall, pad: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    u0, u1, y0, y1 = wall.extent
    base = -wall.d * wall.normal
    corners = np.array([base + wall.u * u + np.array([0.0, y, 0.0]) for u in (u0, u1) for y in (y0, y1)])
    return corners.min(axis=0) - pad, corners.max(axis=0) + pad


def _verify_level(level, segs, supports, room, vlm, frames, root, rng, used, log) -> list[Verified]:
    """Confirms the segments of one level (largest first). Structure level = door / window only."""
    out: list[Verified] = []
    for i, seg in enumerate(segs):
        if i in used:
            continue
        support = supports[i]
        options = candidate_entries(seg, support, level)
        if level == "structure":
            top = max(seg.votes, key=seg.votes.get) if seg.votes else None
            if top not in ("door", "window"):
                continue
            options = [e for e in options if door_window_ok(seg, e, room)]
        if not options:
            continue
        entry = confirm(seg, support, options, _crops(seg, frames, root), vlm, rng)
        if entry is None:
            log(f"objects: verify {level} seg{i} ({seg.size_lwh[2]:.2f} m high) -> rejected by VLM")
            continue
        out.append(Verified(seg, entry, support, confidence_of(seg, entry)))
        used.add(i)
    return out


def run_objects(job_dir: Path, profile: Profile, on_progress: OnProgress) -> int:
    """Builds the v2 object table for a finished job; returns the number of objects (structure included)."""
    from recon3d.capture.format import load_capture
    from recon3d.objects.detect import DownloadError

    job_dir = Path(job_dir)
    workdir = job_dir / "capture"
    out_dir = job_dir / "objects"
    device = _device()
    tracker = vlm = None
    t_start = time.monotonic()
    log = lambda text: _log(job_dir, text)  # noqa: E731
    try:
        # --- inputs ---
        on_progress(0.0, STAGE_STRUCTURE)
        cap = load_capture(job_dir / "capture.zip", workdir)
        frames = cap.frames
        n = len(frames)
        W, H = cap.manifest.image_width, cap.manifest.image_height
        poses_path = job_dir / "poses_cv.npy"
        if poses_path.is_file():
            c2w = np.asarray(np.load(poses_path), dtype=np.float64)[:n]
        else:
            c2w = np.stack([gl_to_cv_c2w(f.pose) for f in frames])
        Ks = np.array([[[f.fx, 0, f.cx], [0, f.fy, f.cy], [0, 0, 1]] for f in frames], dtype=np.float64)
        h, w = round(H * DEPTH_SCALE), round(W * DEPTH_SCALE)
        Ks_small = Ks.copy()
        Ks_small[:, 0, :] *= w / W
        Ks_small[:, 1, :] *= h / H

        # --- structure (0 -> 0.15): depth of every frame, then planes ---
        t0 = time.monotonic()
        depths: list[np.ndarray] = []
        for i, d in enumerate(_render_depths(job_dir / "scene.ply", c2w, Ks, W, H, device=device, scale=DEPTH_SCALE)):
            depths.append(np.asarray(d, dtype=np.float16))
            on_progress(0.12 * (i + 1) / n, STAGE_STRUCTURE)
        cloud = depth_point_cloud(depths, Ks_small, c2w)
        room = fit_room(cloud, c2w[:, :3, 3])
        on_progress(0.15, STAGE_STRUCTURE)
        log(
            f"objects: structure {len(cloud)} points, floor={room.floor_y}, ceiling={room.ceiling_y}, "
            f"{len(room.walls)} walls, {time.monotonic() - t0:.1f}s"
        )

        # --- tracking (0.15 -> 0.55) ---
        t0 = time.monotonic()
        on_progress(0.15, STAGE_TRACK)
        try:
            tracker = _tracker(device, profile)
            paths = [cap.root / f.image for f in frames]
            tracks = tracker.run(paths, (h, w), on_frame=lambda fi: on_progress(0.15 + 0.4 * (fi + 1) / n, STAGE_TRACK))
        except DownloadError as e:
            raise ModelDownloadFailed(str(e)) from e
        resident = profile.name != "laptop" and _free_gb(device) >= profile.objects_resident_min_free_gb
        if not resident:
            tracker.close()
            tracker = None
        segs_raw = [s for t in tracks if (s := track_to_segment(t, depths, Ks_small, c2w, h, w)) is not None]
        segs = sorted(merge_segments(segs_raw), key=lambda s: -s.volume())
        log(
            f"objects: track {n} frames -> {len(tracks)} tracks, {len(segs_raw)} segments, {len(segs)} after merge, "
            f"{time.monotonic() - t0:.1f}s"
        )

        # --- verification (0.55 -> 0.9) ---
        t0 = time.monotonic()
        on_progress(0.55, STAGE_VERIFY)
        supports = {i: support_of(seg, room, segs[:i]) for i, seg in enumerate(segs)}
        rng = random.Random(0)
        try:
            vlm = _vlm(device, profile)
        except DownloadError as e:
            raise ModelDownloadFailed(str(e)) from e
        used: set[int] = set()
        verified: list[Verified] = []
        for li, level in enumerate(("structure", "furniture", "item")):
            try:
                verified += _verify_level(level, segs, supports, room, vlm, frames, cap.root, rng, used, log)
            except DownloadError as e:
                raise ModelDownloadFailed(str(e)) from e
            on_progress(0.55 + 0.3 * (li + 1) / 3, STAGE_VERIFY)
        before = len(verified)
        verified = harmony_filter(verified, room)
        vlm.close()
        vlm = None
        if tracker is not None:
            tracker.close()
            tracker = None
        counts = {lv: sum(v.entry.level == lv for v in verified) for lv in ("structure", "furniture", "item")}
        log(
            f"objects: verify {len(segs)} segments -> {before} confirmed, {len(verified)} after harmony {counts}, "
            f"{time.monotonic() - t0:.1f}s"
        )

        # --- assignment (0.9 -> 0.97) ---
        t0 = time.monotonic()
        on_progress(0.9, STAGE_ASSIGN)
        positions = read_splat_positions(job_dir / "scene.splat")
        # object ids 1..N in verified order (structure openings, furniture, items)
        ids = assign_mod.assign_by_voxels(positions, [v.seg for v in verified])
        openings = [(i + 1, v.seg) for i, v in enumerate(verified) if v.entry.en in ("door", "window")]
        ids, structure_ids = assign_mod.assign_structure(positions, room, ids, first_id=len(verified) + 1, openings=openings)
        structure_kinds: list[tuple[str, Wall | None]] = []
        if room.floor_y is not None:
            structure_kinds.append(("floor", None))
        if room.ceiling_y is not None:
            structure_kinds.append(("ceiling", None))
        structure_kinds += [("wall", wl) for wl in room.walls]
        assert len(structure_kinds) == len(structure_ids)

        # drop anything that owns no Gaussian (cannot be highlighted), renumber 1..M
        counts_g = np.bincount(ids, minlength=len(verified) + len(structure_ids) + 1)
        keep_obj = [i for i in range(len(verified)) if counts_g[i + 1] > 0]
        keep_struct = [k for k, sid in enumerate(structure_ids) if counts_g[sid] > 0]
        remap = np.zeros(len(counts_g), dtype=np.uint16)
        new_id = 1
        obj_new: dict[int, int] = {}
        for i in keep_obj:
            remap[i + 1] = new_id
            obj_new[i] = new_id
            new_id += 1
        for k in keep_struct:
            remap[structure_ids[k]] = new_id
            new_id += 1
        ids = remap[ids]
        dropped = (len(verified) - len(keep_obj)) + (len(structure_ids) - len(keep_struct))
        log(
            f"objects: assign {len(positions)} gaussians, {int((ids > 0).sum())} assigned, {dropped} without gaussians dropped, "
            f"{time.monotonic() - t0:.1f}s"
        )

        # --- records, thumbnails, relations, table (0.97 -> 1.0) ---
        t0 = time.monotonic()
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "thumbs").mkdir(parents=True)
        verified_index = {id(v.seg): i for i, v in enumerate(verified)}  # segment -> position in verified
        records: list[ObjectRecord] = []
        for i in keep_obj:
            v = verified[i]
            oid = obj_new[i]
            support = v.support
            if support.startswith("on:"):  # on:<index into segs> -> on:<final object id>, or none if that object was dropped
                vi = verified_index.get(id(segs[int(support.split(":")[1])]))
                support = f"on:{obj_new[vi]}" if vi is not None and vi in obj_new else "none"
            fi, box = v.seg.best_views[0]
            img = Image.open(cap.root / frames[fi].image).convert("RGB")
            crop_thumbnail(img, box).save(out_dir / "thumbs" / f"{oid}.jpg", quality=85)
            lo, hi = v.seg.bbox_min, v.seg.bbox_max
            records.append(
                ObjectRecord(
                    id=oid, label=v.entry.en, label_zh=v.entry.zh, score=float(v.confidence), views=len(v.seg.frames),
                    center=tuple(float(x) for x in (lo + hi) / 2), bbox_min=tuple(float(x) for x in lo),
                    bbox_max=tuple(float(x) for x in hi), gaussian_count=int((ids == oid).sum()), level=v.entry.level,
                    size=tuple(float(x) for x in v.seg.size_lwh), support=support, confidence=float(v.confidence),
                )
            )
        first_img = Image.open(cap.root / frames[0].image).convert("RGB")
        xz = cloud[:, [0, 2]] if len(cloud) else np.zeros((1, 2))
        for k in keep_struct:
            kind, wall = structure_kinds[k]
            oid = int(remap[structure_ids[k]])
            if kind == "wall":
                lo, hi = _wall_bbox(wall)
            else:
                y = room.floor_y if kind == "floor" else room.ceiling_y
                lo = np.array([xz[:, 0].min(), y - 0.05, xz[:, 1].min()])
                hi = np.array([xz[:, 0].max(), y + 0.05, xz[:, 1].max()])
            crop_thumbnail(first_img, (0, 0, first_img.width, first_img.height)).save(out_dir / "thumbs" / f"{oid}.jpg", quality=85)
            records.append(
                ObjectRecord(
                    id=oid, label=kind, label_zh=STRUCTURE_ZH[kind], score=1.0, views=n,
                    center=tuple(float(x) for x in (lo + hi) / 2), bbox_min=tuple(float(x) for x in lo),
                    bbox_max=tuple(float(x) for x in hi), gaussian_count=int((ids == oid).sum()), level="structure",
                    size=(float(hi[0] - lo[0]), float(hi[2] - lo[2]), float(hi[1] - lo[1])), support="none", confidence=1.0,
                )
            )
        records.sort(key=lambda r: r.id)
        home_look = c2w[0][:3, 2] if len(c2w) else None  # OpenCV +Z = view direction
        relations = compute_relations([r for r in records if r.level != "structure"], home_look)
        write_objects(out_dir, job_dir.name, n, ids, records, relations, structure=room.to_json())
        log(
            f"objects: table {len(records)} records ({len(keep_obj)} objects + {len(keep_struct)} structure), "
            f"thumbs+relations {time.monotonic() - t0:.1f}s; total {time.monotonic() - t_start:.1f}s"
        )
        on_progress(1.0, None)
        return len(records)
    finally:
        for m in (tracker, vlm):
            if m is not None:
                try:
                    m.close()
                except Exception:  # pragma: no cover
                    traceback.print_exc()
        shutil.rmtree(workdir, ignore_errors=True)
