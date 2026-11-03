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
