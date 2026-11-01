import shutil
from dataclasses import replace

import pytest

from recon3d.capture.format import CaptureFormatError
from recon3d.config import PROFILES
from recon3d.jobs.pipeline import run_pipeline
from tests.fixtures.make_capture import make_capture


def test_bad_zip_raises_capture_format_error(tmp_path):
    (tmp_path / "capture.zip").write_bytes(b"not a zip")
    with pytest.raises(CaptureFormatError):
        run_pipeline(tmp_path, PROFILES["laptop"], lambda p: None)
    assert not (tmp_path / "capture").exists()  # extraction dir cleaned up


@pytest.mark.gpu
def test_pipeline_end_to_end(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    shutil.copy(make_capture(tmp_path, n_frames=4), job / "capture.zip")
    profile = replace(PROFILES["laptop"], iterations=200)
    seen = []
    run_pipeline(job, profile, seen.append)
    assert (job / "scene.splat").stat().st_size > 0
    assert (job / "scene.ply").stat().st_size > 0
    assert not (job / "capture").exists()
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert any(abs(p - 0.9) < 1e-9 for p in seen)  # training ends at 0.9, export -> 1.0


def test_gpu_cache_released_after_each_job_oom_then_success(tmp_path, monkeypatch):
    import io
    import time

    import torch

    import recon3d.recon.export_splat as export_mod
    import recon3d.recon.trainer as trainer_mod
    from recon3d.jobs.store import JobStore
    from recon3d.jobs.worker import Worker

    calls = {"train": 0, "empty_cache": 0}

    def fake_train(ts, profile, on_progress, device="cuda"):
        calls["train"] += 1
        if calls["train"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        on_progress(1.0)
        return object()

    def fake_export(splats, path, *a, **k):
        path.write_bytes(b"\0" * 32)
        return 1

    monkeypatch.setattr(trainer_mod, "train", fake_train)
    monkeypatch.setattr(export_mod, "export_splat", fake_export)
    monkeypatch.setattr(export_mod, "export_ply", fake_export)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1))

    store = JobStore(tmp_path / "data")
    w = Worker(store, lambda d, p: run_pipeline(d, PROFILES["laptop"], p))
    w.start()
    try:
        zip_bytes = make_capture(tmp_path, n_frames=2).read_bytes()
        a = store.create(io.BytesIO(zip_bytes))
        b = store.create(io.BytesIO(zip_bytes))
        end = time.monotonic() + 30
        while store.get(b).state not in ("done", "failed") and time.monotonic() < end:
            time.sleep(0.01)
    finally:
        w.stop()
    assert store.get(a).state == "failed" and "显存不足" in store.get(a).message
    assert store.get(b).state == "done"
    assert calls["empty_cache"] == 2  # once per job, failed one included
    assert not (store.job_dir(a) / "capture").exists()
