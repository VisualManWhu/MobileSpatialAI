import numpy as np
import pytest
import torch

from recon3d.recon.dataset import TrainingSet
from recon3d.recon.init_points import init_points


def _ts(depths, n=2, H=40, W=60):
    K = torch.tensor([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]]).repeat(n, 1, 1)
    c2w = torch.eye(4).repeat(n, 1, 1)
    c2w[1, 0, 3] = 0.5
    return TrainingSet(
        images=torch.rand(n, H, W, 3), K=K, c2w=c2w, depths=depths
    )


@pytest.fixture
def ts_plane_2m():
    # depth at half the image resolution, all 2 m
    return _ts([torch.full((20, 30), 2.0) for _ in range(2)])


@pytest.fixture
def ts_no_depth():
    return _ts([None, None])


def test_plane_depth_backprojects_to_plane(ts_plane_2m):
    xyz, rgb = init_points(ts_plane_2m, voxel_size=0.05, min_points=10)
    assert xyz.dtype == np.float32 and rgb.dtype == np.float32
    assert len(xyz) >= 10
    assert np.allclose(xyz[:, 2], 2.0, atol=0.03)
    assert rgb.min() >= 0 and rgb.max() <= 1


def test_empty_depth_falls_back_to_random(ts_no_depth):
    xyz, rgb = init_points(ts_no_depth, min_points=5000)
    assert xyz.shape == (5000, 3) and rgb.shape == (5000, 3)
    assert np.all(np.isfinite(xyz))
    assert np.allclose(rgb, 0.5)
    assert xyz[:, 0].min() >= -1.0 and xyz[:, 0].max() <= 1.5


def test_invalid_and_far_dropped_and_padded():
    d = torch.full((20, 30), 2.0)
    d[:, :15] = 0.0
    d[:, 15:20] = 9.0
    xyz, rgb = init_points(_ts([d, None]), voxel_size=0.05, min_points=500)
    assert xyz.shape == (500, 3)
    assert np.sum(np.isclose(rgb, 0.5).all(axis=1)) >= 1


def test_depth_intrinsics_used_when_present(tmp_path):
    from recon3d.capture.format import load_capture
    from recon3d.config import PROFILES
    from recon3d.recon.dataset import build_training_set
    from tests.fixtures.make_capture import make_capture

    # image 64x48 fx=50 -> scaled depth fx would be 6.25; depth intrinsics say 3.125
    z = make_capture(tmp_path, n_frames=1, poses=[[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
                     depth_intrinsics=(3.125, 3.125, 4.0, 3.0))
    cap = load_capture(z, tmp_path / "w")
    assert cap.frames[0].depth_fx == pytest.approx(3.125)
    ts = build_training_set(cap, PROFILES["laptop"])
    xyz, _ = init_points(ts, voxel_size=0.01, min_points=0)
    assert np.allclose(xyz[:, 2], -1.5, atol=0.01)  # identity GL pose looks down -Z
    # u in 0..7, cx=4, depth 1.5 m: x = (u-4)/3.125*1.5 -> min -1.92 (scaled K would give -0.96)
    assert xyz[:, 0].min() == pytest.approx(-1.92, abs=0.02)
