"""End-to-end: synthetic capture zip -> real JobStore + Worker + pipeline -> scene.splat."""
import io
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from recon3d.config import PROFILES
from recon3d.jobs.pipeline import run_pipeline
from recon3d.jobs.store import JobStore
from recon3d.jobs.worker import Worker
from tests.fixtures.make_capture import make_capture
from tests.fixtures.synthetic import FOCAL_FRAC, make_synthetic_scene

CV_TO_GL = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def cv_c2w_to_arcore_pose(c2w: torch.Tensor) -> list[float]:
    """Inverse of recon3d.capture.poses.gl_to_cv_c2w: right-multiply diag(1,-1,-1,1),
    flatten column-major."""
    return (c2w.cpu() @ CV_TO_GL).T.reshape(-1).tolist()


@pytest.mark.gpu
def test_synthetic_capture_to_splat_via_worker(tmp_path):
    train, _, _ = make_synthetic_scene(n_train=24, n_test=1, device="cuda")
    W, H = train.images.shape[2], train.images.shape[1]
    images = [(im.clamp(0, 1) * 255).round().byte().cpu().numpy() for im in train.images]
    depths = [(d * 1000).clamp(0, 65535).round().to(torch.uint16).cpu().numpy().astype(np.uint16)
              for d in train.depths]
    zip_path = make_capture(
        tmp_path, n_frames=24, images=images, depths=depths,
        fx=FOCAL_FRAC * W, fy=FOCAL_FRAC * W, cx=W / 2, cy=H / 2,
        poses=[cv_c2w_to_arcore_pose(c) for c in train.c2w],
    )

    profile = replace(PROFILES["laptop"], iterations=1000)
    store = JobStore(tmp_path / "data")
    worker = Worker(store, lambda d, p: run_pipeline(d, profile, p))
    worker.start()
    try:
        job_id = store.create(io.BytesIO(zip_path.read_bytes()))
        end = time.monotonic() + 600
        while store.get(job_id).state not in ("done", "failed") and time.monotonic() < end:
            time.sleep(0.2)
    finally:
        worker.stop(30)

    status = store.get(job_id)
    assert status.state == "done", status.message
    size = (store.job_dir(job_id) / "scene.splat").stat().st_size
    assert size > 0 and size % 32 == 0
