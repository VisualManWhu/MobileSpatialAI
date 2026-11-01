"""capture.zip -> trained Gaussians -> scene.splat + scene.ply, inside one job directory."""
from __future__ import annotations

import gc
import shutil
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from recon3d.config import Profile

TRAIN_SHARE = 0.9


def run_pipeline(job_dir: Path, profile: Profile, on_progress: Callable[[float], None]) -> None:
    job_dir = Path(job_dir)
    workdir = job_dir / "capture"
    try:
        _run(job_dir, workdir, profile, on_progress)
    except BaseException as e:
        # The traceback pins _run's frames and thus the dataset/Gaussians (GPU tensors) until
        # the worker is done with the exception; drop the frame locals now, keep line info.
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        # Extracted frames are large and not needed after training.
        shutil.rmtree(workdir, ignore_errors=True)
        _release_gpu_memory()


def _run(job_dir: Path, workdir: Path, profile: Profile, on_progress: Callable[[float], None]) -> None:
    # Heavy imports (torch, gsplat) stay here so the API/worker import without a GPU stack.
    from recon3d.capture.format import load_capture
    from recon3d.recon.dataset import build_training_set
    from recon3d.recon.export_splat import export_ply, export_splat
    from recon3d.recon.trainer import train

    cap = load_capture(job_dir / "capture.zip", workdir)
    ts = build_training_set(cap, profile)
    del cap
    splats = train(ts, profile, lambda p: on_progress(TRAIN_SHARE * p))
    del ts
    on_progress(TRAIN_SHARE)
    export_splat(splats, job_dir / "scene.splat")
    on_progress(0.95)
    export_ply(splats, job_dir / "scene.ply")
    on_progress(1.0)


def _release_gpu_memory() -> None:
    """Drop this job's tensors (incl. those pinned by a failed frame) before the next job."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
