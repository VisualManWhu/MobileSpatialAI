import json
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image

import recon3d.objects.pipeline as op
from recon3d.config import PROFILES
from recon3d.objects.detect import FrameDetection
from recon3d.objects.pipeline import run_objects
from recon3d.recon.export_splat import export_ply, export_splat
from recon3d.recon.trainer import Splats
from tests.fixtures.make_capture import make_capture


def _job(tmp_path, n_frames=6):
    job = tmp_path / "job"
    job.mkdir()
    shutil.copy(make_capture(tmp_path, n_frames=n_frames), job / "capture.zip")
    n = 300
    g = torch.Generator().manual_seed(0)
    means = torch.rand(n, 3, generator=g) * 2 - 1
    means[:, 2] -= 2.0  # in front of the fixture cameras (they look down -Z from z=0)
    s = Splats(
        means=means,
        scales=torch.full((n, 3), float(np.log(0.05))),
        quats=torch.tensor([[1.0, 0, 0, 0]] * n),
        opacities=torch.full((n,), 3.0),
        sh0=torch.zeros(n, 1, 3),
        shN=torch.zeros(n, 0, 3),
    )
    export_splat(s, job / "scene.splat")
    export_ply(s, job / "scene.ply")
    return job


class FakeDetector:
    def __init__(self, dets_by_call):
        self.calls = 0
        self.dets = dets_by_call
        self.closed = False

    def detect(self, image):
        out = self.dets(self.calls, image.size)
        self.calls += 1
        return out

    def close(self):
        self.closed = True


def fake_depths(ply, c2w, Ks, W, H, device="cuda", scale=0.5):
    for _ in range(len(c2w)):
        yield np.full((round(H * scale), round(W * scale)), 2.0, np.float32)


def _install(monkeypatch, dets):
    fake = FakeDetector(dets)
    monkeypatch.setattr(op, "detector_factory", lambda device: fake, raising=False)
    monkeypatch.setattr(op, "render_depths_fn", fake_depths, raising=False)
    return fake


def centre_box(size):
    W, H = size
    m = np.zeros((H, W), bool)
    m[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = True
    return FrameDetection("cup", 0.7, (W / 4, H / 4, 3 * W / 4, 3 * H / 4), m)


def test_pipeline_writes_objects_zip_with_stage_messages(tmp_path, monkeypatch):
    job = _job(tmp_path)
    fake = _install(monkeypatch, lambda i, size: [centre_box(size)])
    seen = []
    count = run_objects(job, PROFILES["laptop"], lambda p, m: seen.append((p, m)))
    assert count == 1 and fake.closed
    z = zipfile.ZipFile(job / "objects.zip")
    assert {"objects.json", "objects.bin", "thumbs/1.jpg"} <= set(z.namelist())
    j = json.loads((job / "objects" / "objects.json").read_text(encoding="utf-8"))
    assert j["objects"][0]["label_zh"] == "杯子" and j["objects"][0]["views"] == 6 and j["frames_used"] == 6
    ids = np.frombuffer((job / "objects" / "objects.bin").read_bytes(), "<u2")
    assert len(ids) == 300 and ids.max() == 1
    assert j["objects"][0]["gaussian_count"] == int((ids == 1).sum()) > 0
    msgs = [m for _, m in seen]
    assert msgs[0] == "识别物体 (检测)"
    assert {"识别物体 (合并)", "识别物体 (归属)", "识别物体 (截图)"} <= set(msgs)
    ps = [p for p, _ in seen]
    assert ps == sorted(ps) and ps[-1] == 1.0
    assert not (job / "capture").exists()
    Image.open(job / "objects" / "thumbs" / "1.jpg").verify()
    log = (job / "log.txt").read_text(encoding="utf-8")
    assert "detect" in log and "assign" in log  # per-stage timings


def test_pipeline_with_no_detections_yields_an_empty_table(tmp_path, monkeypatch):
    job = _job(tmp_path)
    _install(monkeypatch, lambda i, size: [])
    assert run_objects(job, PROFILES["laptop"], lambda p, m: None) == 0
    j = json.loads((job / "objects" / "objects.json").read_text(encoding="utf-8"))
    assert j["objects"] == [] and (job / "objects.zip").is_file()
    assert np.frombuffer((job / "objects" / "objects.bin").read_bytes(), "<u2").max() == 0


def test_pipeline_uses_poses_cv_when_present(tmp_path, monkeypatch):
    job = _job(tmp_path)
    captured = {}

    def spy_depths(ply, c2w, Ks, W, H, device="cuda", scale=0.5):
        captured["c2w"] = np.array(c2w, copy=True)
        yield from fake_depths(ply, c2w, Ks, W, H, device, scale)

    _install(monkeypatch, lambda i, size: [])
    monkeypatch.setattr(op, "render_depths_fn", spy_depths, raising=False)
    poses = np.stack([np.eye(4)] * 6)
    poses[:, 0, 3] = 9.0
    np.save(job / "poses_cv.npy", poses)
    run_objects(job, PROFILES["laptop"], lambda p, m: None)
    assert np.allclose(captured["c2w"][:, 0, 3], 9.0)


def test_frame_sampling_caps_at_240():
    assert op.sample_indices(1123) == list(range(0, 1123, 5)) and len(op.sample_indices(1123)) <= 240
    assert op.sample_indices(100) == list(range(100))
