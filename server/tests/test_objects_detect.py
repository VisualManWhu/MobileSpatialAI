import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from recon3d.objects.detect import Detector, FrameDetection
from recon3d.objects.vocab import VOCAB

OFFICE_ZIP = Path(__file__).resolve().parents[1] / "data" / "jobs" / "40a80bd6c17f" / "capture.zip"


@pytest.fixture(scope="module")
def detector():
    det = Detector()
    yield det
    det.close()


@pytest.mark.gpu
def test_detector_returns_vocab_labels_and_full_size_masks(detector):
    img = Image.new("RGB", (640, 480), (235, 235, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([250, 200, 390, 400], fill=(200, 40, 40))  # a red mug body
    d.arc([380, 240, 440, 360], 270, 90, fill=(200, 40, 40), width=14)  # handle
    out = detector.detect(img)
    assert out, "something should be detected on the drawing"
    vocab = {e.en for e in VOCAB}
    assert all(isinstance(o, FrameDetection) and o.label in vocab for o in out)
    assert all(o.mask.shape == (480, 640) and o.mask.dtype == bool for o in out)
    best = max(out, key=lambda o: o.score)
    assert best.mask[300, 320]  # the SAM mask covers the body centre
    assert best.box[0] < 320 < best.box[2] and best.box[1] < 300 < best.box[3]


@pytest.mark.gpu
@pytest.mark.skipif(not OFFICE_ZIP.is_file(), reason="office capture not on this machine")
def test_detector_recognises_real_office_objects(detector):
    with zipfile.ZipFile(OFFICE_ZIP) as z:
        img = Image.open(io.BytesIO(z.read("images/000466.jpg"))).convert("RGB")
    out = detector.detect(img)
    labels = {o.label for o in out}
    assert len(labels & {"chair", "bottle", "box", "plant", "book", "remote control", "door"}) >= 3
    # NMS: no two kept boxes overlap heavily
    from recon3d.objects.merge import _box_iou_2d

    assert all(_box_iou_2d(a.box, b.box) < 0.7 for i, a in enumerate(out) for b in out[i + 1 :])
