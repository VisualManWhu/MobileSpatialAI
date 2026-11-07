import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import recon3d.recon.dataset as pl_dataset
from recon3d.capture.format import CaptureFormatError, load_capture
from recon3d.capture.poses import gl_to_cv_c2w
from recon3d.config import PROFILES
from recon3d.jobs.pipeline import run_pipeline
from recon3d.recon.dataset import build_training_set
from recon3d.recon.precise_pose import PrecisePoseFailed, PrecisePoseResult
from tests.fixtures.make_capture import make_capture


def test_bad_zip_raises_capture_format_error(tmp_path):
    (tmp_path / "capture.zip").write_bytes(b"not a zip")
    with pytest.raises(CaptureFormatError):
        run_pipeline(tmp_path, PROFILES["laptop"], lambda p, m: None)
    assert not (tmp_path / "capture").exists()  # extraction dir cleaned up


@pytest.mark.gpu
def test_pipeline_end_to_end(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    shutil.copy(make_capture(tmp_path, n_frames=4), job / "capture.zip")
    profile = replace(PROFILES["laptop"], iterations=200)
    seen = []
    assert run_pipeline(job, profile, lambda p, m: seen.append((p, m))) is None
    assert (job / "scene.splat").stat().st_size > 0
    assert (job / "scene.ply").stat().st_size > 0
    assert not (job / "capture").exists()
    ps = [p for p, _ in seen]
    assert ps == sorted(ps) and ps[-1] == 1.0
    assert any(abs(p - 0.9) < 1e-9 for p in ps)  # training ends at 0.9, export -> 1.0
    assert all(m is None for _, m in seen)  # fast mode: worker picks the default stage text


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
    w = Worker(store, lambda d, m, p: run_pipeline(d, PROFILES["laptop"], p, m))
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


def test_android_v2_fixture_trains_with_raw_confidence(tmp_path):
    # Produced by CapturePackageWriterTest.writesV2SampleForServerContract (Android JVM test).
    cap = load_capture(Path(__file__).parent / "fixtures" / "android_v2.zip", tmp_path / "w")
    assert cap.manifest.format_version == 2
    assert [f.depth_kind for f in cap.frames] == ["raw", "smooth"]
    assert cap.frames[0].confidence == "conf/000001.png" and cap.frames[1].confidence is None
    ts = build_training_set(cap, PROFILES["laptop"])
    assert ts.images.shape[0] == 2
    assert all(d is not None for d in ts.depths)
    assert ts.confs[0].shape == ts.depths[0].shape
    assert ts.confs[0].numpy() == pytest.approx(200 / 255)  # constant map from the PNG
    assert ts.confs[1] is not None  # heuristic confidence for smooth depth
    assert not (ts.confs[1].numpy() == pytest.approx(200 / 255))
    # raw depth carries the pose of its own frame (2 deg yaw, 5 cm back), smooth depth does not
    assert cap.frames[0].depth_pose is not None and cap.frames[1].depth_pose is None
    assert not np.allclose(cap.frames[0].depth_pose, cap.frames[0].pose)
    assert cap.frames[0].depth_pose[2, 3] == pytest.approx(0.05)
    assert ts.depth_c2w[0] is not None and ts.depth_c2w[1] is None
    assert float(ts.depth_c2w[0][2, 3]) == pytest.approx(0.05)


def _fake_training(monkeypatch):
    """CPU-only stand-ins for train/export so the pipeline tests need no GPU."""
    import recon3d.recon.export_splat as export_mod
    import recon3d.recon.trainer as trainer_mod

    def fake_train(ts, profile, on_progress, device="cuda", **kw):
        on_progress(0.5)
        on_progress(1.0)
        return object()

    def fake_export(splats, path, *a, **k):
        path.write_bytes(b"\0" * 32)
        return 1

    monkeypatch.setattr(trainer_mod, "train", fake_train)
    monkeypatch.setattr(export_mod, "export_splat", fake_export)
    monkeypatch.setattr(export_mod, "export_ply", fake_export)


def _job_with_capture(tmp_path, n_frames=4):
    job = tmp_path / "job"
    job.mkdir()
    shutil.copy(make_capture(tmp_path, n_frames=n_frames), job / "capture.zip")
    return job


def test_pipeline_fast_mode_unchanged_progress(tmp_path, monkeypatch):
    _fake_training(monkeypatch)
    job = _job_with_capture(tmp_path)
    seen = []
    assert run_pipeline(job, PROFILES["laptop"], lambda p, m: seen.append((p, m))) is None
    ps = [p for p, _ in seen]
    assert ps == sorted(ps) and ps[-1] == 1.0 and any(abs(p - 0.9) < 1e-9 for p in ps)
    assert all(m is None for _, m in seen)


def test_pipeline_precise_uses_refined_poses_and_stage_messages(tmp_path, monkeypatch):
    import recon3d.jobs.pipeline as pl

    _fake_training(monkeypatch)
    job = _job_with_capture(tmp_path)
    captured = {}

    def fake_precise(cap, workdir, on_progress):
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "db.db").write_bytes(b"sift")  # COLMAP leaves ~1 GB here per 1000 frames
        on_progress(0.35)  # real precise_pose's first call: a whole intrinsic group extracted
        on_progress(0.6)
        on_progress(1.0)
        c2w = np.stack([gl_to_cv_c2w(f.pose) for f in cap.frames])
        c2w[:, 0, 3] += 1.0
        return PrecisePoseResult(c2w_cv=c2w, registered=np.ones(4, bool), stats={})

    real_bts = pl_dataset.build_training_set

    def spy_bts(cap, profile, device="cpu", c2w_override=None):
        captured["override"] = None if c2w_override is None else c2w_override.copy()
        return real_bts(cap, profile, device, c2w_override)

    monkeypatch.setattr(pl, "precise_pose", fake_precise, raising=False)  # lazily imported: patch where looked up
    monkeypatch.setattr(pl_dataset, "build_training_set", spy_bts)
    seen = []
    note = run_pipeline(job, replace(PROFILES["laptop"], iterations=50), lambda p, m: seen.append((p, m)), mode="precise")
    assert note is None
    assert captured["override"] is not None and captured["override"][0, 0, 3] != 0
    assert np.allclose(np.load(job / "poses_cv.npy"), captured["override"])  # kept for eval_scene / drift tools
    assert (job / "pose_stats.json").is_file()
    assert seen[0] == (0.0, "位姿优化 (特征提取)")  # extraction is shown from the first second, not 训练中 0%
    msgs = [m for _, m in seen if m]
    assert msgs[0] == "位姿优化 (特征提取)" and "位姿优化 (特征匹配)" in msgs and "位姿优化 (全局优化)" in msgs
    assert not (job / "colmap").exists()  # workdir removed with the job's other scratch
    ps = [p for p, _ in seen]
    assert ps == sorted(ps) and max(p for p, m in seen if m) <= 0.3 + 1e-9
    assert ps[-1] == 1.0 and any(abs(p - 0.9) < 1e-9 for p in ps)


def test_pipeline_precise_falls_back_on_failure(tmp_path, monkeypatch):
    import recon3d.jobs.pipeline as pl

    _fake_training(monkeypatch)
    job = _job_with_capture(tmp_path)

    def fake_precise(cap, workdir, on_progress):
        raise PrecisePoseFailed(0.42)

    monkeypatch.setattr(pl, "precise_pose", fake_precise, raising=False)
    note = run_pipeline(job, PROFILES["laptop"], lambda p, m: None, mode="precise")
    assert note == "完成（高精位姿失败，已按快速模式重建）"
    assert (job / "scene.splat").stat().st_size > 0
    log = (job / "log.txt").read_text(encoding="utf-8")  # pipeline appends the reason itself
    assert "0.42" in log or "42%" in log


def test_precise_import_error_falls_back(tmp_path, monkeypatch):
    _fake_training(monkeypatch)
    job = _job_with_capture(tmp_path)
    monkeypatch.setitem(sys.modules, "recon3d.recon.precise_pose", None)  # import -> ImportError
    note = run_pipeline(job, PROFILES["laptop"], lambda p, m: None, mode="precise")
    assert note == "完成（高精位姿失败，已按快速模式重建）"
    assert "ImportError" in (job / "log.txt").read_text(encoding="utf-8")


def test_pipeline_keeps_colmap_workdir_when_asked(tmp_path, monkeypatch):
    import recon3d.jobs.pipeline as pl

    _fake_training(monkeypatch)
    job = _job_with_capture(tmp_path)

    def fake_precise(cap, workdir, on_progress):
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "db.db").write_bytes(b"sift")
        c2w = np.stack([gl_to_cv_c2w(f.pose) for f in cap.frames])
        return PrecisePoseResult(c2w_cv=c2w, registered=np.ones(4, bool), stats={})

    monkeypatch.setattr(pl, "precise_pose", fake_precise, raising=False)
    monkeypatch.setenv("RECON3D_KEEP_COLMAP", "1")
    run_pipeline(job, PROFILES["laptop"], lambda p, m: None, mode="precise")
    assert (job / "colmap" / "db.db").is_file()
