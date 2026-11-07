import dataclasses

import numpy as np
import pytest
import torch

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


def test_depth_aspect_mismatch_without_intrinsics_skips(tmp_path, caplog):
    import logging

    d = np.full((9, 16), 1500, dtype=np.uint16)  # 16:9 depth vs 4:3 image
    cap = load_capture(make_capture(tmp_path, n_frames=2, depths=[d, d]), tmp_path / "w")
    with caplog.at_level(logging.WARNING):
        ts = build_training_set(cap, PROFILES["laptop"])
    assert all(x is None for x in ts.depths)
    assert "aspect" in caplog.text


def test_depth_aspect_mismatch_with_intrinsics_kept(tmp_path):
    d = np.full((9, 16), 1500, dtype=np.uint16)
    cap = load_capture(make_capture(tmp_path, depths=[d], n_frames=1,
                                    depth_intrinsics=(12.0, 12.0, 8.0, 4.5)), tmp_path / "w")
    ts = build_training_set(cap, PROFILES["laptop"])
    assert ts.depths[0] is not None
    assert ts.depth_K[0][0, 0] == pytest.approx(12.0)
    assert ts.depth_K[0][1, 2] == pytest.approx(4.5)


def test_training_set_confs_from_png(tmp_path):
    conf = [np.full((6, 8), 128, np.uint8)]
    z = make_capture(tmp_path, n_frames=1, format_version=2, confidences=conf,
                     depths=[np.full((6, 8), 1500, np.uint16)])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    assert ts.confs[0].shape == (6, 8)
    assert abs(float(ts.confs[0][0, 0]) - 128 / 255) < 1e-6


def test_training_set_confs_heuristic_for_v1(tmp_path):
    z = make_capture(tmp_path, n_frames=1, depths=[np.full((6, 8), 1500, np.uint16)])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    assert float(ts.confs[0].min()) == 1.0  # flat depth, no edges


def test_depth_c2w_from_depth_pose_in_cv_convention(tmp_path):
    dp = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.3, 0.0, 0.05, 1]
    z = make_capture(tmp_path, n_frames=3, format_version=2, depth_poses=[dp, None, dp], with_depth=True)
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    assert len(ts.depth_c2w) == 3
    assert ts.depth_c2w[1] is None
    d = ts.depth_c2w[0]
    assert d.shape == (4, 4)
    assert float(d[0, 3]) == pytest.approx(0.3) and float(d[2, 3]) == pytest.approx(0.05)
    # OpenGL -> OpenCV flips the y and z axes of the rotation block
    assert float(d[1, 1]) == pytest.approx(-1.0) and float(d[2, 2]) == pytest.approx(-1.0)
    assert not torch.allclose(d, ts.c2w[0])


def test_depth_c2w_none_without_depth(tmp_path):
    dp = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.3, 0.0, 0.05, 1]
    z = make_capture(tmp_path, n_frames=1, format_version=2, with_depth=False, depth_poses=[dp])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    assert ts.depth_c2w == [None]


def test_conf_size_mismatch_skips_depth(tmp_path, caplog):
    conf = [np.full((3, 4), 255, np.uint8)]
    z = make_capture(tmp_path, n_frames=1, format_version=2, confidences=conf,
                     depths=[np.full((6, 8), 1500, np.uint16)])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    assert ts.depths[0] is None and ts.confs[0] is None
    assert "confidence" in caplog.text


def test_c2w_override_replaces_poses_and_corrects_depth_pose(tmp_path):
    dp = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.3, 0.0, 0.05, 1]
    z = make_capture(tmp_path, n_frames=3, format_version=2, depth_poses=[dp, None, dp], with_depth=True)
    cap = load_capture(z, tmp_path / "w")
    base = build_training_set(cap, PROFILES["laptop"])
    override = base.c2w.numpy().astype(np.float64).copy()
    shift = np.eye(4)
    shift[:3, 3] = [1.0, 2.0, 3.0]
    override[0] = shift @ override[0]
    ts = build_training_set(cap, PROFILES["laptop"], c2w_override=override)
    assert torch.allclose(ts.c2w, torch.tensor(override, dtype=torch.float32))
    assert ts.depth_c2w[1] is None
    expected0 = torch.tensor(override[0], dtype=torch.float32) @ torch.linalg.inv(base.c2w[0]) @ base.depth_c2w[0]
    assert torch.allclose(ts.depth_c2w[0], expected0, atol=1e-5)
    assert torch.allclose(ts.depth_c2w[2], base.depth_c2w[2], atol=1e-6)  # frame 2 unchanged


def test_c2w_override_bad_shape_raises(tmp_path):
    z = make_capture(tmp_path, n_frames=2)
    cap = load_capture(z, tmp_path / "w")
    with pytest.raises(ValueError):
        build_training_set(cap, PROFILES["laptop"], c2w_override=np.zeros((3, 4, 4)))
