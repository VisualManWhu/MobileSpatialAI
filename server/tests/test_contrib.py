import math
import time

import pytest
import torch

pytest.importorskip("gsplat")

from recon3d.recon.contrib import compute_contrib, prune_mask  # noqa: E402
from recon3d.recon.init_points import init_points  # noqa: E402
from tests.fixtures.synthetic import make_synthetic_scene  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_ts():
    train_ts, test_ts, _ = make_synthetic_scene(device="cuda")
    return train_ts, test_ts


def _params(means: torch.Tensor) -> dict:
    n = means.shape[0]
    return {
        "means": means,
        "quats": torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(n, 1),
        "scales": torch.full((n, 3), math.log(0.02), device="cuda"),
        "opacities": torch.full((n,), 2.0, device="cuda"),
    }


@pytest.mark.gpu
def test_contrib_counts_visible_views_only(synthetic_ts):
    tr = synthetic_ts[0]
    xyz, rgb = init_points(tr)
    # Per-pixel weights sum to <= 1, so the full ~174k cloud (~14 Gaussians/pixel) never reaches
    # 1 px each; subsample to ~3.5k so the >= 3 views expectation is meaningful.
    full = torch.from_numpy(xyz).cuda().float()
    xyz = xyz[::50]
    n = len(xyz)
    means = torch.cat([torch.from_numpy(xyz).cuda().float(), torch.tensor([[0.0, 1e4, 0.0]], device="cuda")])  # far off the orbit axis: outside every camera FOV (a z=1e4 point is in front of the cameras that look along +z)
    params = _params(means)
    H, W = tr.images.shape[1:3]
    viewmats = torch.linalg.inv(tr.c2w.cuda())
    t0 = time.perf_counter()
    total, nviews = compute_contrib(params, viewmats, tr.K.cuda(), W, H, downscale=2)
    torch.cuda.synchronize()
    print(f"compute_contrib: {len(viewmats)} views, {n + 1} gaussians, {time.perf_counter() - t0:.3f}s")
    fp = _params(full)
    t1 = time.perf_counter()
    compute_contrib(fp, viewmats, tr.K.cuda(), W, H, downscale=2)
    torch.cuda.synchronize()
    print(f"compute_contrib full cloud: {len(viewmats)} views, {len(full)} gaussians, {time.perf_counter() - t1:.3f}s")
    assert total.shape == (n + 1,) and nviews.dtype == torch.int32
    assert nviews[-1] == 0 and total[-1] == 0
    assert (nviews[:-1] >= 3).float().mean() > 0.8


@pytest.mark.gpu
def test_contrib_handles_size_not_divisible_by_downscale(synthetic_ts):
    tr = synthetic_ts[0]
    xyz, _ = init_points(tr)
    params = _params(torch.from_numpy(xyz[:5]).cuda().float())
    viewmats = torch.linalg.inv(tr.c2w.cuda())
    total, nviews = compute_contrib(params, viewmats, tr.K.cuda(), 255, 191, downscale=2)
    assert total.shape == (5,)


def test_prune_mask_thresholds():
    total = torch.tensor([0.1, 5.0, 5.0, 0.6])
    nviews = torch.tensor([10, 2, 3, 3])
    assert prune_mask(total, nviews, min_views=3, min_px=0.5).tolist() == [True, True, False, False]
