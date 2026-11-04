import numpy as np
import torch

from recon3d.recon.bounds import SceneBounds


def _c2w(*centers):
    out = torch.eye(4).repeat(len(centers), 1, 1)
    for i, c in enumerate(centers):
        out[i, :3, 3] = torch.tensor(c, dtype=torch.float32)
    return out


def test_far_radius_uses_nearest_camera():
    b = SceneBounds.from_c2w(_c2w((0, 0, 0), (10, 0, 0)), far_m=5.0, bbox_margin_m=0.0)
    pts = torch.tensor([[1.0, 0, 0], [9.0, 0, 0], [5.0, 0, 0], [5.0, 4.0, 0], [0, 0, 5.1]])
    assert b.keep(pts).tolist() == [True, True, True, False, False]


def test_bbox_margin_grows_camera_box():
    b = SceneBounds.from_c2w(_c2w((0, 0, 0), (2, 2, 0)), far_m=0.0, bbox_margin_m=1.0)
    pts = torch.tensor([[-0.9, 1.0, 0.5], [-1.1, 1.0, 0.0], [3.0, 3.0, 1.0], [1.0, 1.0, 1.01]])
    assert b.keep(pts).tolist() == [True, False, True, False]


def test_both_criteria_apply_and_zero_disables():
    both = SceneBounds.from_c2w(_c2w((0, 0, 0), (2, 0, 0)), far_m=5.0, bbox_margin_m=1.0)
    # inside the sphere of camera 2 but outside the grown box
    assert both.keep(torch.tensor([[4.0, 0.0, 0.0]])).tolist() == [False]
    off = SceneBounds.from_c2w(_c2w((0, 0, 0)), far_m=0.0, bbox_margin_m=0.0)
    assert not off.enabled
    assert off.keep(torch.tensor([[1e3, 0, 0]])).tolist() == [True]


def test_keep_np_and_chunking_agree():
    cams = _c2w((0, 0, 0), (3, 1, 0), (-2, 2, 1))
    b = SceneBounds.from_c2w(cams, far_m=4.0, bbox_margin_m=4.0)
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-8, 8, size=(200_000, 3)).astype(np.float32)
    got = b.keep_np(xyz)
    c = cams[:, :3, 3].numpy()
    d = np.sqrt(((xyz[:, None, :] - c[None]) ** 2).sum(-1)).min(1)
    inbox = ((xyz >= c.min(0) - 4) & (xyz <= c.max(0) + 4)).all(1)
    assert np.array_equal(got, (d <= 4.0) & inbox)
    assert 0 < got.sum() < len(xyz)


def test_support_keeps_near_cameras_or_near_depth_cloud():
    cloud = np.array([[6.0, 0.0, 0.0]], dtype=np.float32)
    b = SceneBounds.from_c2w(
        _c2w((0, 0, 0)), far_m=0.0, bbox_margin_m=0.0,
        support_xyz=cloud, support_far_m=2.0, support_dist_m=0.3,
    )
    assert b.enabled and b.far_m == 0 and b.lo is None
    pts = torch.tensor([
        [1.0, 0.0, 0.0],   # near a camera: kept regardless of the cloud
        [6.1, 0.0, 0.0],   # far from cameras, 0.1 m from a depth point: kept
        [6.0, 0.0, 0.6],   # far from cameras, 0.6 m from the cloud: floater
        [-4.0, 0.0, 0.0],  # far, no cloud anywhere near: floater
    ])
    assert b.keep(pts).tolist() == [True, True, False, False]


def test_support_disabled_by_zero_or_missing_cloud():
    cams = _c2w((0, 0, 0))
    cloud = np.zeros((1, 3), dtype=np.float32)
    for kw in (
        dict(support_xyz=None, support_far_m=2.0, support_dist_m=0.3),
        dict(support_xyz=cloud, support_far_m=0.0, support_dist_m=0.3),
        dict(support_xyz=cloud, support_far_m=2.0, support_dist_m=0.0),
        dict(support_xyz=cloud[:0], support_far_m=2.0, support_dist_m=0.3),
    ):
        b = SceneBounds.from_c2w(cams, far_m=0.0, bbox_margin_m=0.0, **kw)
        assert not b.enabled and b.support_keys is None
        assert b.keep(torch.tensor([[50.0, 0, 0]])).tolist() == [True]


def test_support_agrees_with_brute_force_within_tolerance():
    rng = np.random.default_rng(1)
    cloud = rng.uniform(-5, 5, size=(20_000, 3)).astype(np.float32)
    cams = _c2w((0, 0, 0))
    dist = 0.3
    b = SceneBounds.from_c2w(cams, far_m=0.0, bbox_margin_m=0.0, support_xyz=cloud, support_far_m=1.0, support_dist_m=dist)
    q = rng.uniform(-6, 6, size=(5_000, 3)).astype(np.float32)
    got = b.keep_np(q)
    from scipy.spatial import cKDTree

    d_cloud = cKDTree(cloud).query(q, k=1)[0]
    d_cam = np.linalg.norm(q, axis=1)
    near = d_cam <= 1.0
    # exact within dist must be kept; beyond 2 * sqrt(3) * dist must go (unless near a camera)
    assert got[near | (d_cloud <= dist)].all()
    assert not got[~near & (d_cloud > 2 * np.sqrt(3) * dist)].any()
    assert 0 < got.sum() < len(q)
