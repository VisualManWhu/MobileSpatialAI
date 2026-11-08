"""End-to-end pipeline v2 on a synthetic room: analytic depth of a box room with a chair-sized block
and a phone-sized slab on top of it, a fake tracker (masks from the analytic id map), a fake VLM."""
import json
import shutil
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

import recon3d.objects.pipeline as op
from recon3d.capture.poses import gl_to_cv_c2w
from recon3d.config import PROFILES
from recon3d.objects.pipeline import run_objects
from recon3d.objects.segments import Track
from recon3d.recon.export_splat import export_ply, export_splat
from recon3d.recon.trainer import Splats
from tests.fixtures.make_capture import make_capture

W, H, FX = 384, 216, 160.0
N_FRAMES = 48  # each half of the chair track needs a few down-pitched frames to see its legs 3 times
ROOM_LO, ROOM_HI = np.array([-2.0, 0.0, -2.5]), np.array([2.0, 2.6, 2.5])
CHAIR_LO, CHAIR_HI = np.array([0.75, 0.0, -1.25]), np.array([1.25, 0.9, -0.75])
PHONE_LO, PHONE_HI = np.array([0.9, 0.9, -1.05]), np.array([1.08, 0.92, -0.96])  # 18 x 9 x 2 cm on the chair
TALL_LO, TALL_HI = np.array([-2.0, 0.0, 0.0]), np.array([-1.9, 2.0, 0.9])  # a door-like slab flush with the x = -2 wall


def gl_pose(yaw_deg, pitch_deg, eye=(0.0, 1.4, 0.0)):
    """OpenGL camera-to-world (column-major 16 floats) at `eye` looking along yaw/pitch."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.sin(y) * np.cos(p), np.sin(p), -np.cos(y) * np.cos(p)])  # -Z forward at yaw 0
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, cam_up, -fwd, eye
    return m.T.ravel().tolist()  # column-major


# Even frames look towards the chair corner (yaw 45), odd frames towards the x = -2 wall with the door (yaw -90);
# pitch cycles level / down / up so floor and ceiling are both seen. Small yaw jitter so views differ.
POSES = [gl_pose((45 if i % 2 == 0 else -90) + 6 * ((i // 2) % 5 - 2), (5, -25, 10)[i % 3]) for i in range(N_FRAMES)]


def _ray_box(eye, dirs, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lo - eye) / dirs
        t1 = (hi - eye) / dirs
    tmin = np.max(np.minimum(t0, t1), axis=-1)
    tmax = np.min(np.maximum(t0, t1), axis=-1)
    return tmin, tmax


def render(c2w_cv, K, w, h, objects):
    """Analytic depth (camera +Z) and object-id map: 0 = room shell, 1.. = objects in order."""
    vs, us = np.mgrid[0:h, 0:w]
    dirs_cam = np.stack([(us + 0.5 - K[0, 2]) / K[0, 0], (vs + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(us, float)], -1)
    dirs = dirs_cam @ c2w_cv[:3, :3].T
    eye = c2w_cv[:3, 3]
    _, t_exit = _ray_box(eye, dirs, ROOM_LO, ROOM_HI)  # camera is inside the room: hit = exit
    t = t_exit.copy()
    ids = np.zeros((h, w), np.int32)
    for i, (lo, hi) in enumerate(objects, start=1):
        tmin, tmax = _ray_box(eye, dirs, lo, hi)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < t)
        t[hit] = tmin[hit]
        ids[hit] = i
    return (t * dirs_cam[..., 2]).astype(np.float32), ids


class Scene:
    def __init__(self, objects):
        self.objects = objects

    def depths(self, ply, c2w, Ks, width, height, device="cuda", scale=0.5):
        h, w = round(height * scale), round(width * scale)
        for c, K in zip(c2w, Ks):
            Ks_small = K.copy()
            Ks_small[0] *= w / width
            Ks_small[1] *= h / height
            yield render(c, Ks_small, w, h, self.objects)[0]

    def ids(self, frame, c2w, Ks, width, height, scale=0.5):
        h, w = round(height * scale), round(width * scale)
        K = Ks[frame].copy()
        K[0] *= w / width
        K[1] *= h / height
        return render(c2w[frame], K, w, h, self.objects)[1]


class FakeTracker:
    """Tracks straight from the analytic id map; `splits` maps object index -> list of frame ranges."""

    def __init__(self, scene, c2w, Ks, splits, votes):
        self.scene, self.c2w, self.Ks = scene, c2w, Ks
        self.splits, self.votes = splits, votes
        self.closed = False
        self.frames_seen = 0

    def run(self, image_paths, mask_hw, on_frame=None):
        h, w = mask_hw
        sx, sy = W / w, H / h
        tracks = []
        tid = 1
        for oi, ranges in self.splits.items():
            for rng in ranges:
                t = Track(tid, [], [], [], dict(self.votes[oi]))
                for fi in rng:
                    m = self.scene.ids(fi, self.c2w, self.Ks, W, H) == oi + 1
                    if m.sum() < 16:
                        continue
                    ys, xs = np.nonzero(m)
                    t.frames.append(fi)
                    t.masks.append(np.packbits(m))
                    t.boxes.append((xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy))
                if t.frames:
                    tracks.append(t)
                tid += 1
        for fi in range(len(image_paths)):
            self.frames_seen += 1
            if on_frame:
                on_frame(fi)
        return tracks

    def close(self):
        self.closed = True


class FakeVlm:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.closed = False

    def ask(self, images, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else "不确定"

    def close(self):
        self.closed = True


def _job(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    shutil.copy(make_capture(tmp_path, n_frames=N_FRAMES, image_width=W, image_height=H, fx=FX, fy=FX, poses=POSES), job / "capture.zip")
    # Gaussians: a grid filling the room shell neighbourhood, the chair and the phone
    g = torch.Generator().manual_seed(0)
    pts = [torch.rand(3000, 3, generator=g) * torch.tensor(ROOM_HI - ROOM_LO) + torch.tensor(ROOM_LO)]
    pts[0][:, 1] = torch.where(torch.rand(3000, generator=g) < 0.5, torch.full((3000,), 0.01), pts[0][:, 1])  # half on the floor
    pts.append(torch.rand(400, 3, generator=g) * torch.tensor(CHAIR_HI - CHAIR_LO) + torch.tensor(CHAIR_LO))
    pts.append(torch.rand(60, 3, generator=g) * torch.tensor(PHONE_HI - PHONE_LO) + torch.tensor(PHONE_LO))
    pts.append(torch.rand(300, 3, generator=g) * torch.tensor(TALL_HI - TALL_LO) + torch.tensor(TALL_LO))
    means = torch.cat(pts).float()
    n = len(means)
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


def _geometry():
    c2w = np.stack([gl_to_cv_c2w(np.array(p).reshape(4, 4).T) for p in POSES])
    Ks = np.array([[[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]]] * N_FRAMES, dtype=np.float64)
    return c2w, Ks


def _install(monkeypatch, scene, splits, votes, answers):
    c2w, Ks = _geometry()
    tracker = FakeTracker(scene, c2w, Ks, splits, votes)
    vlm = FakeVlm(answers)
    monkeypatch.setattr(op, "render_depths_fn", scene.depths, raising=False)
    monkeypatch.setattr(op, "tracker_factory", lambda device, profile: tracker, raising=False)
    monkeypatch.setattr(op, "vlm_factory", lambda device, profile: vlm, raising=False)
    return tracker, vlm


def test_pipeline_builds_structure_furniture_and_items(tmp_path, monkeypatch):
    job = _job(tmp_path)
    scene = Scene([(CHAIR_LO, CHAIR_HI), (PHONE_LO, PHONE_HI)])
    # the chair is seen as two tracks (disjoint frames) that must merge; the phone as one track
    tracker, vlm = _install(
        monkeypatch, scene,
        splits={0: [range(0, 24), range(24, 48)], 1: [range(0, 48)]},
        votes={0: {"chair": 4.0, "stool": 1.0}, 1: {"phone": 2.0, "remote control": 1.0}},
        answers=["椅子", "椅子", "手机", "手机"],
    )
    seen = []
    count = run_objects(job, PROFILES["laptop"], lambda p, m: seen.append((p, m)))
    assert tracker.closed and vlm.closed and tracker.frames_seen == N_FRAMES
    j = json.loads((job / "objects" / "objects.json").read_text(encoding="utf-8"))
    assert j["version"] == 2 and j["frames_used"] == N_FRAMES
    by_label = {o["label"]: o for o in j["objects"]}
    assert set(by_label) >= {"chair", "phone", "floor", "ceiling", "wall"}
    assert count == len(j["objects"])
    n_walls = len(j["structure"]["walls"])
    assert 2 <= n_walls <= 4 and sum(o["label"] == "wall" for o in j["objects"]) == n_walls
    chair, phone = by_label["chair"], by_label["phone"]
    assert chair["level"] == "furniture" and chair["support"] == "floor"
    assert chair["views"] == N_FRAMES // 2  # the even (chair-side) frames of both merged tracks
    assert abs(chair["size"][2] - 0.9) < 0.1 and abs(chair["size"][0] - 0.5) < 0.1
    assert abs(chair["confidence"] - 0.8) < 1e-6
    assert phone["level"] == "item" and phone["support"] == f"on:{chair['id']}"
    for o in j["objects"]:
        assert o["gaussian_count"] > 0
    ids = np.frombuffer((job / "objects" / "objects.bin").read_bytes(), "<u2")
    assert ids.max() == len(j["objects"]) and set(np.unique(ids)) - {0} == {o["id"] for o in j["objects"]}
    assert j["structure"]["floor_y"] is not None and abs(j["structure"]["floor_y"]) < 0.05
    assert abs(j["structure"]["ceiling_y"] - 2.6) < 0.05
    # four asks: chair twice, phone twice, with the size and the place in the prompt
    assert len(vlm.prompts) == 4 and "放在地面上" in vlm.prompts[0] and "放在家具上" in vlm.prompts[2]
    msgs = [m for _, m in seen if m]
    stages = [s for i, s in enumerate(msgs) if i == 0 or msgs[i - 1] != s]
    assert stages == ["识别物体 (结构)", "识别物体 (跟踪)", "识别物体 (确认)", "识别物体 (归属)"]
    ps = [p for p, _ in seen]
    assert ps == sorted(ps) and ps[-1] == 1.0
    assert {"objects.json", "objects.bin", f"thumbs/{chair['id']}.jpg"} <= set(zipfile.ZipFile(job / "objects.zip").namelist())
    Image.open(job / "objects" / "thumbs" / f"{chair['id']}.jpg").verify()
    assert not (job / "capture").exists()
    log = (job / "log.txt").read_text(encoding="utf-8")
    assert "structure" in log and "track" in log and "verify" in log and "assign" in log


def test_white_door_labelled_cabinet_is_not_in_the_table(tmp_path, monkeypatch):
    job = _job(tmp_path)
    scene = Scene([(TALL_LO, TALL_HI)])
    tracker, vlm = _install(monkeypatch, scene, splits={0: [range(0, N_FRAMES)]}, votes={0: {"cabinet": 3.0}}, answers=["柜子", "柜子"])
    run_objects(job, PROFILES["laptop"], lambda p, m: None)
    j = json.loads((job / "objects" / "objects.json").read_text(encoding="utf-8"))
    assert all(o["level"] == "structure" for o in j["objects"])
    assert "cabinet" not in {o["label"] for o in j["objects"]}


def test_same_slab_voted_door_becomes_a_door(tmp_path, monkeypatch):
    job = _job(tmp_path)
    scene = Scene([(TALL_LO, TALL_HI)])
    tracker, vlm = _install(monkeypatch, scene, splits={0: [range(0, N_FRAMES)]}, votes={0: {"door": 3.0}}, answers=["门", "门"])
    run_objects(job, PROFILES["laptop"], lambda p, m: None)
    j = json.loads((job / "objects" / "objects.json").read_text(encoding="utf-8"))
    door = [o for o in j["objects"] if o["label"] == "door"]
    assert len(door) == 1 and door[0]["level"] == "structure"
    assert door[0]["support"] == "floor" or door[0]["support"].startswith("wall:")  # floor wins the support priority
    assert door[0]["gaussian_count"] > 0


def test_unsure_vlm_drops_the_object(tmp_path, monkeypatch):
    job = _job(tmp_path)
    scene = Scene([(CHAIR_LO, CHAIR_HI)])
    _install(monkeypatch, scene, splits={0: [range(0, N_FRAMES)]}, votes={0: {"chair": 1.0}}, answers=["椅子", "不确定"])
    run_objects(job, PROFILES["laptop"], lambda p, m: None)
    j = json.loads((job / "objects" / "objects.json").read_text(encoding="utf-8"))
    assert "chair" not in {o["label"] for o in j["objects"]}


def test_merge_module_is_gone():
    with pytest.raises(ImportError):
        import recon3d.objects.merge  # noqa: F401


def test_profiles_carry_objects_models():
    lap, srv = PROFILES["laptop"], PROFILES["server"]
    assert lap.objects_detector == "IDEA-Research/grounding-dino-tiny" and lap.objects_sam == "facebook/sam2.1-hiera-small"
    assert lap.objects_vlm == "Qwen/Qwen2.5-VL-3B-Instruct" and lap.objects_vlm_4bit is True
    assert srv.objects_detector == "IDEA-Research/grounding-dino-base" and srv.objects_sam == "facebook/sam2.1-hiera-large"
    assert srv.objects_vlm == "Qwen/Qwen2.5-VL-7B-Instruct" and srv.objects_vlm_4bit is False
    assert srv.objects_resident_min_free_gb == 30.0
