import numpy as np
import pytest
import torch

from recon3d.capture.format import load_capture
from recon3d.config import PROFILES
from recon3d.recon.confidence import depth_confidence
from recon3d.recon.dataset import TrainingSet, build_training_set
from recon3d.recon.init_points import init_points
from tests.fixtures.make_capture import make_capture


def _ts(depths, n=2, H=40, W=60):
    K = torch.tensor([[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]]).repeat(n, 1, 1)
    c2w = torch.eye(4).repeat(n, 1, 1)
    c2w[1, 0, 3] = 0.5
    return TrainingSet(
        images=torch.rand(n, H, W, 3), K=K, c2w=c2w, depths=depths,
        confs=[None if d is None else torch.from_numpy(depth_confidence(d.numpy(), None)) for d in depths],
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
    # without padding only the 2 m region survives: the 9 m columns are dropped
    real, _ = init_points(_ts([d, None]), voxel_size=0.05, min_points=0)
    assert len(real) > 0
    assert np.abs(real[:, 2]).max() < 5.0


def test_depth_intrinsics_used_when_present(tmp_path):
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


def test_init_points_drops_low_confidence(tmp_path):
    depth = np.full((6, 8), 1500, np.uint16)
    conf = np.full((6, 8), 255, np.uint8); conf[:, :4] = 0
    z = make_capture(tmp_path, n_frames=1, format_version=2, depths=[depth], confidences=[conf])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    xyz, _ = init_points(ts, voxel_size=0.001, min_points=0)
    # right half only: x in camera frame >= 0 for u >= cx (cx = dw/2 scaled)
    assert len(xyz) == 6 * 4
    assert (xyz[:, 0] >= -1e-6).all()


def test_init_points_uses_depth_c2w_when_present():
    ts = _ts([torch.full((20, 30), 2.0), None])
    base, _ = init_points(ts, voxel_size=0.01, min_points=0)
    # Depth captured 0.3 m to the right of (and 0.1 m behind) camera 0: points shift with it.
    dc2w = torch.eye(4)
    dc2w[0, 3] = 0.3
    dc2w[2, 3] = -0.1
    ts.depth_c2w = [dc2w, None]
    xyz, rgb = init_points(ts, voxel_size=0.01, min_points=0)
    assert len(xyz) == len(base)
    assert np.allclose(np.sort(xyz[:, 0]) - np.sort(base[:, 0]), 0.3, atol=0.02)
    assert np.allclose(xyz[:, 2], 1.9, atol=0.03)
    assert rgb.min() >= 0 and rgb.max() <= 1


def test_init_points_colour_sampled_by_projection_with_depth_c2w():
    # Image 0 is black on the left half and white on the right; the depth plane is 2 m away.
    ts = _ts([torch.full((20, 30), 2.0), None])
    img = torch.zeros(40, 60, 3)
    img[:, 30:] = 1.0
    ts.images[0] = img
    xyz0, rgb0 = init_points(ts, voxel_size=0.01, min_points=0)
    assert abs(float((rgb0[:, 0] > 0.5).mean()) - 0.5) < 0.1
    # Depth from a pose 1 m to the right: back-projected points move right, so when projected
    # into image 0 almost all of them land on the white half.
    dc2w = torch.eye(4)
    dc2w[0, 3] = 1.0
    ts.depth_c2w = [dc2w, None]
    xyz, rgb = init_points(ts, voxel_size=0.01, min_points=0)
    assert np.allclose(xyz[:, 0].min(), xyz0[:, 0].min() + 1.0, atol=0.02)
    assert float((rgb[:, 0] > 0.5).mean()) > 0.75


def test_init_points_v1_drops_far_and_edges(tmp_path):
    depth = np.full((6, 8), 1500, np.uint16); depth[:, 6:] = 5000  # 5 m: beyond 4 m, plus an edge at col 5|6
    z = make_capture(tmp_path, n_frames=1, depths=[depth])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    xyz, _ = init_points(ts, voxel_size=0.001, min_points=0)
    assert len(xyz) == 6 * 5  # columns 0..4 survive
