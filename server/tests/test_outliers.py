import numpy as np

from recon3d.recon.outliers import sor_mask


def test_sor_flags_lone_far_point():
    rng = np.random.default_rng(0)
    cloud = rng.normal(0, 0.05, size=(2000, 3)).astype(np.float32)
    xyz = np.concatenate([cloud, [[3.0, 0, 0]]]).astype(np.float32)
    out = sor_mask(xyz, k=8, sigma=3.0)
    assert out[-1] and out[:-1].mean() < 0.02


def test_sor_noop_when_too_few_or_disabled():
    xyz = np.zeros((5, 3), np.float32)
    assert not sor_mask(xyz, k=8).any()
    assert not sor_mask(np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32), sigma=0.0).any()
