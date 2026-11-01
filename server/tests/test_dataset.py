import dataclasses

import numpy as np
import pytest

from recon3d.capture.format import load_capture
from recon3d.config import PROFILES
from recon3d.recon.dataset import build_training_set
from tests.fixtures.make_capture import make_capture


def test_downscale_scales_intrinsics(tmp_path):
    cap = load_capture(make_capture(tmp_path, image_width=1920, image_height=1080, fx=1500), tmp_path / "w")
    ts = build_training_set(cap, PROFILES["laptop"])
    assert ts.images.shape[2] == 960
    assert ts.images.shape[1] == 540
    assert ts.K[0, 0, 0] == pytest.approx(750)
    assert ts.K[0, 0, 2] == pytest.approx(480)
    assert ts.c2w.shape == (4, 4, 4)
    assert 0.0 <= float(ts.images.min()) and float(ts.images.max()) <= 1.0


def test_no_upscale_small_images(tmp_path):
    cap = load_capture(make_capture(tmp_path), tmp_path / "w")
    ts = build_training_set(cap, PROFILES["laptop"])
    assert ts.images.shape[1:3] == (48, 64)
    assert ts.K[0, 0, 0] == pytest.approx(50)


def test_depths_meters_and_none(tmp_path):
    cap = load_capture(make_capture(tmp_path), tmp_path / "w")
    ts = build_training_set(cap, PROFILES["laptop"])
    assert ts.depths[0].shape == (6, 8)
    assert float(ts.depths[0][0, 0]) == pytest.approx(1.5)
    cap2 = load_capture(make_capture(tmp_path, with_depth=False, name="b.zip"), tmp_path / "w2")
    assert all(d is None for d in build_training_set(cap2, PROFILES["laptop"]).depths)


def test_profile_frozen_replace():
    p = dataclasses.replace(PROFILES["laptop"], iterations=10)
    assert p.iterations == 10 and PROFILES["laptop"].iterations == 15000
    with pytest.raises(dataclasses.FrozenInstanceError):
        PROFILES["server"].iterations = 1
