import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import recon3d.objects.track as tr
from recon3d.objects.segments import unpack_mask
from recon3d.objects.track import Tracker, iou_xyxy

OFFICE_ZIP = Path(__file__).resolve().parents[1] / "data" / "jobs" / "40a80bd6c17f" / "capture.zip"
MASK_HW = (540, 960)


def test_iou_xyxy():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert abs(iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not OFFICE_ZIP.is_file(), reason="office capture not on this machine")
def test_tracker_follows_office_objects_over_sixty_frames(tmp_path):
    paths = []
    with zipfile.ZipFile(OFFICE_ZIP) as z:
        names = sorted(n for n in z.namelist() if n.startswith("images/") and n.endswith(".jpg"))[:60]
        for n in names:
            p = tmp_path / Path(n).name
            p.write_bytes(z.read(n))
            paths.append(p)
    W, H = Image.open(paths[0]).size
    seen = []
    tracker = Tracker("cuda", "IDEA-Research/grounding-dino-tiny", "facebook/sam2.1-hiera-small")
    try:
        tracks = tracker.run(paths, MASK_HW, on_frame=seen.append)
    finally:
        tracker.close()
    assert seen == list(range(60))
    long_tracks = [t for t in tracks if len(t.frames) >= 5]
    assert len(long_tracks) >= 3
    for t in tracks:
        assert t.frames == sorted(t.frames) and len(set(t.frames)) == len(t.frames)
        assert len(t.masks) == len(t.frames) == len(t.boxes)
        m = unpack_mask(t.masks[0], *MASK_HW)
        assert m.any()
        x0, y0, x1, y1 = t.boxes[0]
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
        assert t.votes and all(v > 0 for v in t.votes.values())
    labels = {lbl for t in long_tracks for lbl in t.votes}
    assert labels & {"monitor", "laptop", "keyboard", "chair", "bottle", "box", "plant", "book", "door"}


# --- CPU harness: a scripted SAM 2 session + proposer, to pin the bookkeeping around the real model ---


class _Out:
    def __init__(self, pred_masks):
        self.pred_masks = pred_masks


class FakeSession:
    def __init__(self):
        self.processed = 0  # frames fed through forward(), like len(processed_frames)
        self.obj_ids: list[int] = []
        self.seeds: list[tuple] = []  # (kind, frame_idx, obj_ids, processed_at_seed)

    def obj_idx_to_id(self, i):
        return self.obj_ids[i]


class FakeProc:
    def __init__(self):
        self.sessions: list[FakeSession] = []

    def init_video_session(self, **kw):
        s = FakeSession()
        self.sessions.append(s)
        return s

    def add_inputs_to_inference_session(self, session, frame_idx, obj_ids, input_boxes=None, input_masks=None, original_size=None):
        kind = "box" if input_boxes is not None else "mask"
        for o in obj_ids:
            if o not in session.obj_ids:
                session.obj_ids.append(o)
        session.seeds.append((kind, frame_idx, list(obj_ids), session.processed))

    def __call__(self, images=None, device=None, return_tensors=None):
        class _In:
            pixel_values = [None]

        return _In()

    def post_process_masks(self, masks, original_sizes, binarize=True):
        return masks


class FakeModel:
    """mask_fn(obj_id, frame_index_in_run) -> bool mask or None (lost)."""

    def __init__(self, mask_fn, hw):
        self.mask_fn, self.hw = mask_fn, hw
        self.frame = -1  # advanced by the tracker through on_frame hooks below
        self.forward_frames: list[int] = []

    def __call__(self, inference_session, frame=None):
        import torch

        inference_session.processed += 1
        self.forward_frames.append(self.frame)
        masks = []
        for oid in inference_session.obj_ids:
            m = self.mask_fn(oid, self.frame)
            masks.append(torch.from_numpy((m if m is not None else np.zeros(self.hw, bool)).astype(np.float32)))
        return _Out(torch.stack(masks) if masks else torch.zeros((0,) + self.hw))


class FakeDetector:
    def __init__(self, boxes_by_frame):
        self.boxes_by_frame = boxes_by_frame
        self.frame = -1

    def detect_boxes(self, image):
        return self.boxes_by_frame.get(self.frame, [])

    def close(self):
        pass


def _harness(tmp_path, n_frames, boxes_by_frame, mask_fn, hw=(54, 96)):
    import torch

    paths = []
    for i in range(n_frames):
        p = tmp_path / f"{i:04d}.png"
        Image.new("RGB", (192, 108), "white").save(p)
        paths.append(p)
    t = Tracker("cpu", "x", "y")
    t.detector = FakeDetector(boxes_by_frame)
    t._proc = FakeProc()
    t._model = FakeModel(mask_fn, hw)
    t._torch = torch
    # the tracker opens each image before anything else; piggy-back the frame counter on that
    real_open = Image.open

    def counting_open(path, *a, **k):
        idx = int(Path(path).stem)
        t._model.frame = idx
        t.detector.frame = idx
        return real_open(path, *a, **k)

    return t, paths, counting_open


def _blob(hw, x0, y0, x1, y1):
    m = np.zeros(hw, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_seed_frame_index_follows_frames_actually_fed_to_the_session(tmp_path, monkeypatch):
    """Frames with no live track are not fed to SAM 2, so a later seed must use the session's own
    frame count, not the position in the capture (review finding #1)."""
    hw = (54, 96)
    t, paths, opener = _harness(tmp_path, 15, {10: [("cup", 0.6, (40.0, 20.0, 80.0, 60.0))]}, lambda oid, f: _blob(hw, 20, 10, 40, 30))
    monkeypatch.setattr(Image, "open", opener)
    tracks = t.run(paths, hw)
    s = t._proc.sessions[0]
    kind, frame_idx, ids, processed_at_seed = s.seeds[0]
    assert kind == "box" and ids == [1]
    assert frame_idx == processed_at_seed == 0  # nothing had been fed before frame 10
    assert t._model.forward_frames == list(range(10, 15))
    assert tracks[0].frames == list(range(10, 15))


def test_session_reset_reseeds_only_tracks_seen_in_the_previous_frame(tmp_path, monkeypatch):
    """A track already lost for a few frames must end at the reset, not be re-seeded from a stale mask
    (review finding #2)."""
    hw = (54, 96)
    monkeypatch.setattr(tr, "RESET_EVERY", 12)
    boxes = {0: [("cup", 0.6, (10.0, 10.0, 40.0, 40.0)), ("bottle", 0.6, (120.0, 10.0, 160.0, 60.0))]}

    def mask_fn(oid, f):
        if oid == 1:
            return _blob(hw, 5, 5, 20, 20)
        return _blob(hw, 60, 5, 80, 30) if f < 5 else None  # object 2 vanishes from frame 5

    t, paths, opener = _harness(tmp_path, 20, boxes, mask_fn)
    monkeypatch.setattr(Image, "open", opener)
    tracks = t.run(paths, hw)
    assert len(t._proc.sessions) == 2
    reseed = [s for s in t._proc.sessions[1].seeds if s[0] == "mask"]
    assert len(reseed) == 1 and reseed[0][2] == [1]  # only the track that was visible last frame
    by_id = {x.id: x for x in tracks}
    assert by_id[2].frames == list(range(0, 5))
    assert by_id[1].frames == list(range(0, 20))


def test_overlapping_proposals_in_one_frame_make_a_single_track(tmp_path, monkeypatch):
    """Two proposal boxes of the same frame with IoU >= 0.5 are one object: the second only votes
    (review finding #3)."""
    hw = (54, 96)
    boxes = {0: [("cup", 0.6, (40.0, 20.0, 80.0, 60.0)), ("bottle", 0.5, (44.0, 20.0, 84.0, 60.0))]}
    t, paths, opener = _harness(tmp_path, 6, boxes, lambda oid, f: _blob(hw, 20, 10, 40, 30))
    monkeypatch.setattr(Image, "open", opener)
    tracks = t.run(paths, hw)
    assert len(tracks) == 1
    assert tracks[0].votes == {"cup": 0.6, "bottle": 0.5}
