"""capture.zip -> trained Gaussians -> scene.splat + scene.ply, inside one job directory."""
from __future__ import annotations

import gc
import os
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from recon3d.config import Profile

# Progress layout. Fast: training 0 -> TRAIN_SHARE, export -> 1.0.
# Precise: pose refinement 0 -> POSE_SHARE, training POSE_SHARE -> TRAIN_SHARE, export -> 1.0.
POSE_SHARE = 0.3
TRAIN_SHARE = 0.9

STAGE_EXTRACT = "位姿优化 (特征提取)"
STAGE_MATCH = "位姿优化 (特征匹配)"
STAGE_MAPPER = "位姿优化 (全局优化)"
FALLBACK_NOTE = "完成（高精位姿失败，已按快速模式重建）"

# (progress 0..1, stage message or None = the worker's default text by progress)
OnProgress = Callable[[float, "str | None"], None]

# Resolved lazily on first precise job (pycolmap is an optional, heavy import);
# tests inject a fake here.
precise_pose = None


def run_pipeline(job_dir: Path, profile: Profile, on_progress: OnProgress, mode: str = "fast") -> str | None:
    """Run one job. Returns the final status message to use instead of 完成 (e.g. the
    precise-mode fallback note), or None."""
    job_dir = Path(job_dir)
    workdir = job_dir / "capture"
    try:
        return _run(job_dir, workdir, profile, on_progress, mode)
    except BaseException as e:
        # The traceback pins _run's frames and thus the dataset/Gaussians (GPU tensors) until
        # the worker is done with the exception; drop the frame locals now, keep line info.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        # Extracted frames are large and not needed after training.
        shutil.rmtree(workdir, ignore_errors=True)
        _release_gpu_memory()


def _run(job_dir: Path, workdir: Path, profile: Profile, on_progress: OnProgress, mode: str) -> str | None:
    # Heavy imports (torch, gsplat) stay here so the API/worker import without a GPU stack.
    from recon3d.capture.format import load_capture
    from recon3d.recon import dataset as dataset_mod
    from recon3d.recon.export_splat import export_ply, export_splat
    from recon3d.recon.trainer import train

    cap = load_capture(job_dir / "capture.zip", workdir)
    note: str | None = None
    c2w_override = None
    train_start = 0.0
    if mode == "precise":
        train_start = POSE_SHARE
        try:
            result = _precise_pose_fn()(
                cap, job_dir / "colmap", lambda q: on_progress(POSE_SHARE * q, _pose_stage(q))
            )
            c2w_override = result.c2w_cv
        except Exception as e:
            # PrecisePoseFailed (too few frames registered), ImportError (no pycolmap) or a
            # pycolmap crash: rebuild with the ARCore poses and say so in the final message.
            reason = (
                "precise pose unavailable: ImportError (is pycolmap installed in the server venv?)"
                if isinstance(e, ImportError) else "precise pose failed"
            )
            _log(job_dir, f"{reason}, falling back to fast mode:\n" + traceback.format_exc().rstrip())
            note = FALLBACK_NOTE
        on_progress(POSE_SHARE, None)
    ts = dataset_mod.build_training_set(cap, profile, c2w_override=c2w_override)
    del cap
    # Debug dump of the floater-pruned Gaussians (scripts/eval_scene.py --dump-pruned).
    extra = {"pruned_out": job_dir / "pruned.ply"} if os.environ.get("RECON3D_DUMP_PRUNED") == "1" else {}
    span = TRAIN_SHARE - train_start
    splats = train(ts, profile, lambda p: on_progress(min(TRAIN_SHARE, train_start + span * p), None), **extra)
    del ts
    on_progress(TRAIN_SHARE, None)
    export_splat(splats, job_dir / "scene.splat")
    on_progress(0.95, None)
    export_ply(splats, job_dir / "scene.ply")
    on_progress(1.0, None)
    return note


def _pose_stage(q: float) -> str:
    if q < 0.35:
        return STAGE_EXTRACT
    if q < 0.7:
        return STAGE_MATCH
    return STAGE_MAPPER


def _precise_pose_fn():
    if precise_pose is not None:
        return precise_pose
    from recon3d.recon.precise_pose import precise_pose as fn

    return fn


def _log(job_dir: Path, text: str) -> None:
    """Append to the job's log.txt in JobStore.log's format (the pipeline has no store)."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(job_dir / "log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {text}\n")


def _release_gpu_memory() -> None:
    """Drop this job's tensors (incl. those pinned by a failed frame) before the next job."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
