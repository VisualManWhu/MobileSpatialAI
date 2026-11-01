import dataclasses

import pytest
import torch

pytest.importorskip("gsplat")

from recon3d.config import PROFILES  # noqa: E402
from recon3d.recon.init_points import init_points  # noqa: E402
from recon3d.recon.trainer import Splats, _enforce_cap, render, train  # noqa: E402
from tests.fixtures.synthetic import make_synthetic_scene  # noqa: E402


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred.to(gt.device).clamp(0, 1) - gt) ** 2)
    return float(-10.0 * torch.log10(mse))


@pytest.fixture(scope="module")
def synthetic_ts():
    train_ts, test_ts, _ = make_synthetic_scene(device="cuda")
    return train_ts, test_ts


@pytest.mark.gpu
def test_synthetic_scene_is_sane(synthetic_ts):
    train_ts, test_ts = synthetic_ts
    assert train_ts.images.shape == (24, 192, 256, 3)
    assert test_ts.images.shape == (4, 192, 256, 3)
    assert len(train_ts.depths) == 24
    valid = train_ts.depths[0] > 0
    assert 0.05 < valid.float().mean() < 0.95  # sphere covers part of the frame
    assert torch.allclose(train_ts.depths[0][valid].min(), torch.tensor(3.0, device="cuda"), atol=0.2)


@pytest.mark.gpu
def test_trains_synthetic_scene(synthetic_ts):
    train_ts, test_ts = synthetic_ts
    splats = train(train_ts, dataclasses.replace(PROFILES["laptop"], iterations=2000), on_progress=lambda p: None)
    assert isinstance(splats, Splats)
    K = (PROFILES["laptop"].sh_degree + 1) ** 2 - 1
    n = splats.means.shape[0]
    assert splats.scales.shape == (n, 3) and splats.quats.shape == (n, 4)
    assert splats.opacities.shape == (n,)
    assert splats.sh0.shape == (n, 1, 3) and splats.shN.shape == (n, K, 3)
    assert psnr(render(splats, test_ts), test_ts.images) > 25.0


@pytest.mark.gpu
def test_respects_max_gaussians(synthetic_ts):
    p = dataclasses.replace(PROFILES["laptop"], iterations=2000, max_gaussians=3000)
    assert train(synthetic_ts[0], p, on_progress=lambda p: None).means.shape[0] <= 3000


@pytest.mark.gpu
def test_densification_growth_is_capped(synthetic_ts):
    # Sparse (8x subsampled) depth -> ~5.4k init points that densification grows
    # (to ~5.9k uncapped), so the cap must be enforced during training, not at init.
    ts = dataclasses.replace(synthetic_ts[0], depths=[d[::8, ::8].contiguous() for d in synthetic_ts[0].depths])
    n_init = len(init_points(ts)[0])
    cap = n_init + 200
    p = dataclasses.replace(PROFILES["laptop"], iterations=1000, max_gaussians=cap)
    n = train(ts, p, on_progress=lambda p: None).means.shape[0]
    assert n_init < n <= cap


@pytest.mark.gpu
def test_enforce_cap_prunes_lowest_opacity():
    from gsplat import DefaultStrategy

    n = 100
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.randn(n, 3, device="cuda")),
            "scales": torch.nn.Parameter(torch.zeros(n, 3, device="cuda")),
            "quats": torch.nn.Parameter(torch.rand(n, 4, device="cuda")),
            "opacities": torch.nn.Parameter(torch.linspace(-3, 3, n, device="cuda")),
        }
    )
    opts = {k: torch.optim.Adam([{"params": v, "name": k}]) for k, v in params.items()}
    state = DefaultStrategy().initialize_state()
    _enforce_cap(params, opts, state, 60)
    assert len(params["opacities"]) == 60
    assert torch.allclose(params["opacities"], torch.linspace(-3, 3, n, device="cuda")[40:])


@pytest.mark.gpu
def test_image_stack_not_resident_on_gpu(synthetic_ts):
    # 16x the train views (384 frames) kept on CPU; only one frame per step may go to GPU.
    tr = synthetic_ts[0]
    reps = 16
    ts = dataclasses.replace(
        tr,
        images=tr.images.cpu().repeat(reps, 1, 1, 1),
        K=tr.K.cpu().repeat(reps, 1, 1),
        c2w=tr.c2w.cpu().repeat(reps, 1, 1),
        depths=list(tr.depths) + [None] * (len(tr.depths) * (reps - 1)),
    )
    stack_bytes = ts.images.numel() * 4
    p = dataclasses.replace(PROFILES["laptop"], iterations=200, max_gaussians=20000)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    train(ts, p, on_progress=lambda p: None)
    peak = torch.cuda.max_memory_allocated() - base
    assert peak < stack_bytes / 2, (peak, stack_bytes)
    assert ts.images.device.type == "cpu"


@pytest.mark.gpu
def test_opacity_reset_runs(synthetic_ts, monkeypatch):
    import gsplat.strategy.ops as ops

    from recon3d.recon import trainer as trainer_mod

    seen = []
    orig = ops.reset_opa

    def spy(params, optimizers, state, value):
        orig(params=params, optimizers=optimizers, state=state, value=value)
        seen.append((value, float(torch.sigmoid(params["opacities"]).max())))

    monkeypatch.setattr(ops, "reset_opa", spy)
    monkeypatch.setattr(trainer_mod, "RESET_EVERY", 100)
    p = dataclasses.replace(PROFILES["laptop"], iterations=600, max_gaussians=20000)
    train(synthetic_ts[0], p, on_progress=lambda p: None)
    # refine_stop = 300 -> resets at steps 100 and 200
    assert len(seen) == 2
    for value, max_opa in seen:
        assert max_opa <= value + 1e-4


@pytest.mark.gpu
def test_pruning_continues_at_cap(synthetic_ts, monkeypatch):
    from gsplat import DefaultStrategy

    pruned = []
    orig = DefaultStrategy._prune_gs

    def spy(self, params, optimizers, state, step):
        n = orig(self, params, optimizers, state, step)
        pruned.append(n)
        return n

    monkeypatch.setattr(DefaultStrategy, "_prune_gs", spy)
    # No depth -> 5000 random init points, subsampled to exactly the cap; many float in
    # empty space and must still be pruned although the count starts at the cap.
    ts = dataclasses.replace(synthetic_ts[0], depths=[None] * len(synthetic_ts[0].depths))
    p = dataclasses.replace(PROFILES["laptop"], iterations=1000, max_gaussians=3000)
    n = train(ts, p, on_progress=lambda p: None).means.shape[0]
    assert sum(pruned) > 0
    assert n <= 3000


@pytest.mark.gpu
def test_reports_progress_every_100_steps(synthetic_ts):
    seen = []
    p = dataclasses.replace(PROFILES["laptop"], iterations=300, sh_degree=0)
    splats = train(synthetic_ts[0], p, on_progress=seen.append)
    assert seen == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert splats.shN.shape[1] == 0
