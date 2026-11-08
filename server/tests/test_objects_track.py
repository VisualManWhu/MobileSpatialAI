import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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
        m = np.unpackbits(t.masks[0])[: MASK_HW[0] * MASK_HW[1]].reshape(MASK_HW)
        assert m.any()
        x0, y0, x1, y1 = t.boxes[0]
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
        assert t.votes and all(v > 0 for v in t.votes.values())
    labels = {lbl for t in long_tracks for lbl in t.votes}
    assert labels & {"monitor", "laptop", "keyboard", "chair", "bottle", "box", "plant", "book", "door"}
