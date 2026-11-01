"""End-to-end (CPU) check of the pose/axis chain: Android GL pose + depth -> world points.

Pins handedness and up direction: ARCore world is +Y up, the GL camera looks down -Z,
and CV image v grows downward.
"""
import numpy as np
import pytest

from recon3d.capture.format import load_capture
from recon3d.config import PROFILES
from recon3d.recon.dataset import build_training_set
from recon3d.recon.init_points import init_points
from tests.fixtures.make_capture import make_capture

# Column-major GL world_from_camera, camera at origin, pitch 0.
# Yaw 90 deg right: forward (-Z_cam) = +X world, right (+X_cam) = +Z world, up = +Y.
YAW_RIGHT = [0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1]
# Yaw 90 deg left: forward = -X world, right = -Z world.
YAW_LEFT = [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]


def _depth():
    d = np.full((12, 16), 2000, dtype=np.uint16)
    d[3:, 12:] = 1800  # right columns (large u)
    d[:3, :] = 1500  # top rows (small v)
    return d


@pytest.mark.parametrize("pose,fwd,right_sign", [(YAW_RIGHT, 1.0, 1.0), (YAW_LEFT, -1.0, -1.0)])
def test_android_pose_chain_handedness_and_up(tmp_path, pose, fwd, right_sign):
    z = make_capture(tmp_path, n_frames=1, image_width=64, image_height=48,
                     fx=40.0, fy=40.0, poses=[pose], depths=[_depth()])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    xyz, _ = init_points(ts, voxel_size=0.01, min_points=0)
    x, y, zz = xyz[:, 0].astype(np.float64), xyz[:, 1], xyz[:, 2]

    centre = (np.abs(y) < 0.3) & (np.abs(zz) < 0.3)
    assert centre.any()
    assert np.all(fwd * x[centre] > 1.8)  # centre pixel lands near (fwd*2, 0, 0)

    top = np.abs(np.abs(x) - 1.5) < 0.05
    plane2 = np.abs(np.abs(x) - 2.0) < 0.05
    assert top.any() and plane2.any()
    assert y[top].min() > 0  # top rows are above the camera (ARCore +Y up)
    assert y[plane2].min() < 0  # bottom rows are below
    assert y[top].mean() > y[plane2].mean()

    right = np.abs(np.abs(x) - 1.8) < 0.05
    assert right.any()
    assert np.all(right_sign * zz[right] > 0)  # image right = camera right
