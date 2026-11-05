import dataclasses
import logging
import re

import pytest
import torch

pytest.importorskip("gsplat")

from recon3d.config import PROFILES  # noqa: E402
from recon3d.recon.init_points import init_points  # noqa: E402
from recon3d.recon.trainer import Splats, _enforce_cap, reg_loss, render, train  # noqa: E402
from tests.fixtures.synthetic import make_synthetic_scene  # noqa: E402


# The pre-A/B defaults (DefaultStrategy, contribution pruning and SOR on). Tests of the default-strategy
# mechanics (opacity reset, _enforce_cap, pruning at the cap) and of the pruning passes pin these explicitly
# now that the profiles default to MCMC with pruning off.
_DEFAULT_STRATEGY = dict(strategy="default")
_PRUNING_ON = dict(contrib_min_views=3, contrib_min_px=0.5, sor_sigma=3.0)


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
def test_reg_loss_values():
    params = {"opacities": torch.zeros(4, device="cuda"), "scales": torch.zeros(4, 3, device="cuda")}
    assert torch.isclose(reg_loss(params, 0.01, 0.01), torch.tensor(0.01 * 0.5 + 0.01 * 1.0, device="cuda"))
    assert reg_loss(params, 0.0, 0.0).item() == 0.0


@pytest.mark.gpu
def test_respects_max_gaussians(synthetic_ts):
    p = dataclasses.replace(PROFILES["laptop"], iterations=2000, max_gaussians=3000)
    assert train(synthetic_ts[0], p, on_progress=lambda p: None).means.shape[0] <= 3000


@pytest.mark.gpu
def test_densification_growth_is_capped(synthetic_ts):
    # Sparse (8x subsampled) depth -> ~5.4k init points that densification grows
    # (to ~5.9k uncapped), so the cap must be enforced during training, not at init.
    ts = dataclasses.replace(
        synthetic_ts[0],
        depths=[d[::8, ::8].contiguous() for d in synthetic_ts[0].depths],
        confs=[c[::8, ::8].contiguous() for c in synthetic_ts[0].confs],
    )
    n_init = len(init_points(ts)[0])
    cap = n_init + 200
    # Floater pruning shrinks the set below n_init; off here so only growth vs. the cap is measured.
    # DefaultStrategy: growth past the cap is trimmed by _enforce_cap (MCMC caps itself).
    p = dataclasses.replace(PROFILES["laptop"], iterations=1000, max_gaussians=cap, **_DEFAULT_STRATEGY,
                            contrib_min_views=0, contrib_min_px=0.0, sor_sigma=0.0)
    n = train(ts, p, on_progress=lambda p: None).means.shape[0]
    assert n_init < n <= cap


def test_depth_loss_pools_and_weights():
    from recon3d.recon.trainer import depth_loss
    rendered = torch.full((1, 6, 8), 2.0)
    target = torch.full((3, 4), 2.0); target[0, 0] = 3.0   # 1 m error at one cell
    conf = torch.ones(3, 4); conf[0, 1] = 0.0
    l = depth_loss(rendered, target, conf, delta=0.1)
    # Huber(1.0, 0.1) = 0.1*(1.0 - 0.05) = 0.095 ; mean over the 11 cells with conf>0
    assert abs(float(l) - 0.095 / 11) < 1e-6


def test_depth_loss_none_when_not_divisible():
    from recon3d.recon.trainer import depth_loss
    assert depth_loss(torch.zeros(1, 7, 8), torch.zeros(3, 4), torch.ones(3, 4)) is None


def _depth_ts(depth_K=None, dh=48, dw=64, H=192, W=256, confs=True):
    from recon3d.recon.dataset import TrainingSet
    K = torch.tensor([[200.0, 0, 128.0], [0, 200.0, 96.0], [0, 0, 1]])[None]
    return TrainingSet(
        images=torch.zeros(1, H, W, 3),
        K=K,
        c2w=torch.eye(4)[None],
        depths=[torch.ones(dh, dw)],
        depth_K=None if depth_K is None else [depth_K],
        confs=[torch.ones(dh, dw)] if confs else [None],
    )


def _scaled_K(sx=0.25, sy=0.25, fx_mul=1.0):
    return torch.tensor([[200.0 * sx * fx_mul, 0, 128.0 * sx], [0, 200.0 * sy, 96.0 * sy], [0, 0, 1]])


def test_depth_loss_usable_without_depth_K():
    from recon3d.recon.trainer import _depth_loss_usable
    assert _depth_loss_usable(_depth_ts(), 0, 192, 256)
    ts = _depth_ts()
    ts.depth_K = [None]
    assert _depth_loss_usable(ts, 0, 192, 256)


def test_depth_loss_usable_with_scaled_depth_K():
    from recon3d.recon.trainer import _depth_loss_usable
    assert _depth_loss_usable(_depth_ts(depth_K=_scaled_K()), 0, 192, 256)


def test_depth_loss_not_usable_with_mismatched_depth_K():
    from recon3d.recon.trainer import _depth_loss_usable
    assert not _depth_loss_usable(_depth_ts(depth_K=_scaled_K(fx_mul=1.05)), 0, 192, 256)


def test_depth_loss_not_usable_when_not_divisible():
    from recon3d.recon.trainer import _depth_loss_usable
    assert not _depth_loss_usable(_depth_ts(dh=50, dw=64), 0, 192, 256)


def test_depth_loss_not_usable_without_confs():
    from recon3d.recon.trainer import _depth_loss_usable
    assert not _depth_loss_usable(_depth_ts(confs=False), 0, 192, 256)


def _with_depth_view(ts, shift=0.1):
    d = torch.eye(4)
    d[0, 3] = shift
    ts.depth_c2w = [d]
    return ts


def test_depth_view_only_when_pose_differs():
    from recon3d.recon.trainer import _depth_view
    assert _depth_view(_depth_ts(), 0) is None  # no depth_c2w
    ts = _depth_ts()
    ts.depth_c2w = [torch.eye(4)]  # equal to c2w -> treated as absent
    assert _depth_view(ts, 0) is None
    assert _depth_view(_with_depth_view(_depth_ts()), 0) is not None


def test_depth_loss_usable_with_depth_view_relaxes_size_and_K():
    from recon3d.recon.trainer import _depth_loss_usable
    # Non-divisible depth size and a mismatched depth_K are both fine with an own depth pose...
    assert _depth_loss_usable(_with_depth_view(_depth_ts(dh=50, dw=64)), 0, 192, 256)
    assert _depth_loss_usable(_with_depth_view(_depth_ts(depth_K=_scaled_K(fx_mul=1.05))), 0, 192, 256)
    # ...but depth and confidence are still required.
    assert not _depth_loss_usable(_with_depth_view(_depth_ts(confs=False)), 0, 192, 256)


def test_depth_K_falls_back_to_scaled_image_K():
    from recon3d.recon.trainer import _depth_K
    k = _depth_K(_depth_ts(), 0, 192, 256)
    assert torch.allclose(k, _scaled_K())
    dk = _scaled_K(fx_mul=1.05)
    assert torch.equal(_depth_K(_depth_ts(depth_K=dk), 0, 192, 256), dk)


@pytest.mark.gpu
def test_trains_with_depth_loss_and_pose_refine(synthetic_ts, caplog):
    # Depth at 1/4 image resolution so the loss exercises the avg-pool path.
    tr = synthetic_ts[0]
    ts = dataclasses.replace(
        tr,
        depths=[d[::4, ::4].contiguous() for d in tr.depths],
        confs=[c[::4, ::4].contiguous() for c in tr.confs],
    )
    p = dataclasses.replace(PROFILES["laptop"], iterations=300, depth_lambda=0.05, pose_lr=1e-4)
    with caplog.at_level(logging.INFO, logger="recon3d.recon.trainer"):
        splats = train(ts, p, on_progress=lambda p: None)
    assert splats.means.shape[0] > 0
    assert any("位姿修正" in r.getMessage() for r in caplog.records)
    assert any("旋转中位数" in r.getMessage() for r in caplog.records)
    n_depth = sum(d is not None for d in ts.depths)
    assert any(f"深度约束用于 {n_depth}/{n_depth} 个有深度的帧" in r.getMessage() for r in caplog.records)
    assert not any("深度约束跳过" in r.getMessage() for r in caplog.records)
    legacy = dataclasses.replace(PROFILES["laptop"], iterations=300, depth_lambda=0.0, pose_lr=0.0)
    assert train(ts, legacy, on_progress=lambda p: None).means.shape[0] > 0


@pytest.mark.gpu
def test_trains_with_depth_from_own_pose(synthetic_ts, caplog):
    """Raw-depth scenario: frame i carries the depth map of frame i+1 together with that frame's
    pose (depth_c2w). Depth at 1/4 resolution, cropped to a non-divisible 47x63 with explicit
    depth intrinsics, so only the depth-view render path can use it."""
    import time

    tr = synthetic_ts[0]
    n = len(tr.c2w)
    nxt = [(i + 1) % n for i in range(n)]
    dK = tr.K[0].clone()
    dK[:2] *= 0.25
    ts = dataclasses.replace(
        tr,
        depths=[tr.depths[j][::4, ::4][:47, :63].contiguous() for j in nxt],
        confs=[tr.confs[j][::4, ::4][:47, :63].contiguous() for j in nxt],
        depth_K=[dK.clone() for _ in range(n)],
        depth_c2w=[tr.c2w[j].clone() for j in nxt],
    )
    p = dataclasses.replace(PROFILES["laptop"], iterations=2000, depth_lambda=0.05, pose_lr=1e-4)
    with caplog.at_level(logging.INFO, logger="recon3d.recon.trainer"):
        t0 = time.monotonic()
        splats = train(ts, p, on_progress=lambda p: None)
        dt_view = time.monotonic() - t0
    assert any(f"深度约束用于 {n}/{n} 个有深度的帧（其中 {n} 帧按深度自身位姿单独渲染）" in r.getMessage()
               for r in caplog.records)
    assert not any("深度约束跳过" in r.getMessage() for r in caplog.records)
    assert any("训练耗时" in r.getMessage() for r in caplog.records)
    assert psnr(render(splats, synthetic_ts[1]), synthetic_ts[1].images) > 25.0
    # Cost check: the extra depth-only render must stay well under 2x the pooled path.
    pooled = dataclasses.replace(
        tr,
        depths=[d[::4, ::4].contiguous() for d in tr.depths],
        confs=[c[::4, ::4].contiguous() for c in tr.confs],
    )
    t0 = time.monotonic()
    train(pooled, dataclasses.replace(p, pose_lr=0.0), on_progress=lambda p: None)
    dt_pooled = time.monotonic() - t0
    print(f"\ndepth-view path {dt_view:.1f} s vs pooled path {dt_pooled:.1f} s for 2000 steps")
    assert dt_view < 2.0 * dt_pooled + 5.0


@pytest.mark.gpu
def test_pose_refine_receives_gradients(synthetic_ts):
    from recon3d.recon.pose_opt import PoseRefine
    from recon3d.recon.trainer import _init_params, _rasterize

    tr = synthetic_ts[0]
    params = _init_params(tr, PROFILES["laptop"], "cuda")
    pose = PoseRefine(len(tr.c2w), "cuda")
    viewmats = torch.linalg.inv(tr.c2w)
    H, W = tr.images.shape[1:3]
    idx = torch.tensor([3], device="cuda")
    out, _, _ = _rasterize(params, pose(viewmats[3:4], idx), tr.K[3:4], W, H, 0, render_mode="RGB+ED")
    assert out.shape == (1, H, W, 4)
    ((out[..., :3] - tr.images[3:4]).abs().mean() + out[..., 3].mean()).backward()
    g = pose.delta.grad
    assert g is not None and g[3].abs().sum() > 0 and g[0].abs().sum() == 0


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
def test_cull_outside_removes_far_gaussians_only():
    from gsplat import DefaultStrategy

    from recon3d.recon.bounds import SceneBounds
    from recon3d.recon.trainer import _cull_outside

    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[1, 0, 3] = 2.0
    bounds = SceneBounds.from_c2w(c2w, far_m=3.0, bbox_margin_m=3.0)
    means = torch.tensor([[0.0, 0, 0], [4.0, 0, 0], [0, 0, 3.5], [1.0, 1.0, 1.0]], device="cuda")
    n = len(means)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(torch.zeros(n, 3, device="cuda")),
            "quats": torch.nn.Parameter(torch.rand(n, 4, device="cuda")),
            "opacities": torch.nn.Parameter(torch.arange(n, device="cuda", dtype=torch.float32)),
        }
    )
    opts = {k: torch.optim.Adam([{"params": v, "name": k}]) for k, v in params.items()}
    state = DefaultStrategy().initialize_state()
    assert _cull_outside(params, opts, state, bounds) == 1  # only (0,0,3.5) is > 3 m from both cameras
    assert params["opacities"].tolist() == [0.0, 1.0, 3.0]
    assert _cull_outside(params, opts, state, None) == 0
    # never empties the set: nothing is within 1 mm of a camera at z = 5
    far_cams = c2w.clone()
    far_cams[:, 2, 3] = 5.0
    tiny = SceneBounds.from_c2w(far_cams, far_m=1e-3, bbox_margin_m=0.0)
    assert _cull_outside(params, opts, state, tiny) == 0
    assert len(params["means"]) == 3


@pytest.mark.gpu
def test_training_culls_gaussians_outside_bounds(synthetic_ts, caplog):
    ts = synthetic_ts[0]
    p = dataclasses.replace(PROFILES["laptop"], iterations=300, max_gaussians=20000, far_m=1e-3, bbox_margin_m=0.0)
    # far_m this small keeps only Gaussians sitting on a camera: the cull must still never empty the set
    with caplog.at_level(logging.INFO, logger="recon3d.recon.trainer"):
        splats = train(ts, p, on_progress=lambda p: None)
    assert splats.means.shape[0] > 0
    assert any("场景范围裁剪" in r.getMessage() for r in caplog.records)


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
        confs=list(tr.confs) + [None] * (len(tr.confs) * (reps - 1)),
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
    p = dataclasses.replace(PROFILES["laptop"], iterations=600, max_gaussians=20000, **_DEFAULT_STRATEGY)
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
    p = dataclasses.replace(PROFILES["laptop"], iterations=1000, max_gaussians=3000, **_DEFAULT_STRATEGY, **_PRUNING_ON)
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


_CONTRIB_RE = re.compile(r"贡献度剪枝（步 (\d+)）：移除 (\d+) 个高斯")
_CONTRIB_SPLIT_RE = re.compile(r"贡献度剪枝（步 \d+）：移除 (\d+) 个高斯.*；其中仅因视角数 (\d+)、仅因贡献 (\d+)、两者 (\d+)")
_SOR_RE = re.compile(r"离群点滤波（k=8，[\d.]+σ）：移除 (\d+) 个高斯")


@pytest.mark.gpu
def test_training_prunes_by_contribution_and_logs(synthetic_ts, caplog, tmp_path):
    ts = synthetic_ts[0]
    p = dataclasses.replace(PROFILES["laptop"], iterations=400, max_gaussians=20000, **_DEFAULT_STRATEGY, **_PRUNING_ON)
    with caplog.at_level(logging.INFO, logger="recon3d.recon.trainer"):
        splats = train(ts, p, on_progress=lambda p: None, pruned_out=tmp_path / "pruned.ply")
    msgs = [r.getMessage() for r in caplog.records]
    contrib = [m for msg in msgs if (m := _CONTRIB_RE.search(msg))]
    sor = [m for msg in msgs if (m := _SOR_RE.search(msg))]
    # steps iters//2, 3*iters//4 and once after the loop
    assert [int(m.group(1)) for m in contrib] == [200, 300, 400]
    splits = [m for msg in msgs if (m := _CONTRIB_SPLIT_RE.search(msg))]
    assert len(splits) == 3
    for m in splits:
        assert int(m.group(1)) == sum(int(m.group(k)) for k in (2, 3, 4))
    assert len(sor) == 1
    assert splats.means.shape[0] > 0
    removed = sum(int(m.group(2)) for m in contrib) + int(sor[0].group(1))
    assert (tmp_path / "pruned.ply").exists() == (removed > 0)
    if removed > 0:
        from recon3d.recon.export_splat import load_ply

        assert load_ply(tmp_path / "pruned.ply").means.shape[0] == removed


@pytest.mark.gpu
def test_contribution_prune_never_empties(synthetic_ts):
    ts = synthetic_ts[0]
    p = dataclasses.replace(PROFILES["laptop"], iterations=400, max_gaussians=20000,
                            **_DEFAULT_STRATEGY, **{**_PRUNING_ON, "contrib_min_views": 10_000})
    assert train(ts, p, on_progress=lambda p: None).means.shape[0] > 0


@pytest.mark.gpu
def test_pruning_off_matches_old_behaviour(synthetic_ts, caplog, tmp_path):
    ts = synthetic_ts[0]
    p = dataclasses.replace(PROFILES["laptop"], iterations=300, max_gaussians=20000, strategy="default",
                            contrib_min_views=0, contrib_min_px=0.0, sor_sigma=0.0,
                            opacity_reg=0.0, scale_reg=0.0)
    with caplog.at_level(logging.INFO, logger="recon3d.recon.trainer"):
        n = train(ts, p, on_progress=lambda p: None, pruned_out=tmp_path / "pruned.ply").means.shape[0]
    assert n > 0
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("贡献度剪枝" in m or "离群点滤波" in m for m in msgs)
    assert not (tmp_path / "pruned.ply").exists()


@pytest.mark.gpu
def test_trains_with_mcmc_within_cap(synthetic_ts):
    tr, te = synthetic_ts
    p = dataclasses.replace(PROFILES["laptop"], iterations=1500, max_gaussians=30000, strategy="mcmc")
    splats = train(tr, p, on_progress=lambda p: None)
    assert 0 < splats.means.shape[0] <= 30000
    assert psnr(render(splats, te), te.images) > 20.0


@pytest.mark.gpu
def test_mcmc_contrib_prune_lowers_opacity_instead_of_removing(synthetic_ts):
    from gsplat import MCMCStrategy

    from recon3d.recon.trainer import _prune_contrib

    ts = synthetic_ts[0]
    xyz, rgb = init_points(ts)
    xyz, rgb = torch.from_numpy(xyz[:2000]).cuda(), torch.from_numpy(rgb[:2000]).cuda()
    # one Gaussian far outside every camera's view: nviews 0, so it is marked
    means = torch.cat([xyz, torch.tensor([[0.0, 1e4, 0.0]], device="cuda")])
    n = len(means)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(torch.full((n, 3), -4.0, device="cuda")),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(n, 1)),
            "opacities": torch.nn.Parameter(torch.zeros(n, device="cuda")),
            "sh0": torch.nn.Parameter(torch.cat([rgb, rgb[:1]])[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros(n, 3, 3, device="cuda")),
        }
    )
    opts = {k: torch.optim.Adam([{"params": v, "name": k}]) for k, v in params.items()}
    state = MCMCStrategy().initialize_state()
    p = dataclasses.replace(PROFILES["laptop"], strategy="mcmc", **_PRUNING_ON)
    viewmats = torch.linalg.inv(ts.c2w.cuda())
    H, W = ts.images.shape[1:3]
    n_marked, marked, split = _prune_contrib(params, opts, state, viewmats, ts.K.cuda(), W, H, p, relocatable=True)
    assert n_marked >= 1
    assert len(params["means"]) == n
    assert marked is not None and marked.means.shape[0] == n_marked
    assert marked.means.device.type == "cpu" and sum(split) == n_marked
    assert float(torch.sigmoid(marked.opacities).min()) == pytest.approx(0.5)  # sliced before lowering
    assert float(torch.sigmoid(params["opacities"][-1])) < 1e-3
    # after refine_stop_iter (relocatable=False) the same profile removes; the trainer passes an
    # empty state under MCMC because ops.remove would slice MCMC's "binoms" table
    n_removed, removed, _ = _prune_contrib(params, opts, {}, viewmats, ts.K.cuda(), W, H, p, relocatable=False)
    assert n_removed >= 1
    assert len(params["means"]) == n - n_removed
    assert removed is not None and removed.means.shape[0] == n_removed
    assert not (params["means"][:, 1] > 1e3).any()


def test_profile_defaults_mcmc_without_pruning():
    for name in ("laptop", "server"):
        p = PROFILES[name]
        assert p.strategy == "mcmc"
        assert p.contrib_min_views == 0 and p.contrib_min_px == 0.0 and p.sor_sigma == 0.0
        assert p.opacity_reg == 0.01 and p.scale_reg == 0.01
        assert p.far_m == 5.0 and p.bbox_margin_m == 5.0 and p.contrib_view_px == 1.0
        assert p.support_far_m == 2.0 and p.support_dist_m == 0.3
        assert p.appearance_lr == 0.0 and p.appearance_reg == 1e-2
        assert p.mono_lambda == 0.1 and p.mono_lambda_end == 0.01 and p.mono_patch == 64
        assert p.mcmc_noise_lr == 1e4 and p.max_scale_m == 0.0


def test_cull_huge_removes_oversized_only():
    from recon3d.recon.trainer import _cull_huge

    n = 6
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.zeros(n, 3)),
        "scales": torch.nn.Parameter(torch.log(torch.tensor([[0.1, 0.1, 0.1]] * 5 + [[0.05, 1.5, 0.05]]))),
        "quats": torch.nn.Parameter(torch.zeros(n, 4)),
        "opacities": torch.nn.Parameter(torch.zeros(n)),
        "sh0": torch.nn.Parameter(torch.zeros(n, 1, 3)),
        "shN": torch.nn.Parameter(torch.zeros(n, 3, 3)),
    })
    opts = {k: torch.optim.Adam([v]) for k, v in params.items()}
    assert _cull_huge(params, opts, {}, 0.0) == 0 and len(params["means"]) == n
    assert _cull_huge(params, opts, {}, 1.0) == 1 and len(params["means"]) == n - 1
    assert float(torch.exp(params["scales"]).max()) <= 1.0


@pytest.mark.gpu
def test_trains_with_mono_depth_prior(synthetic_ts, caplog, monkeypatch):
    import recon3d.recon.trainer as tr

    # Stand-in for Depth Anything: the synthetic scene's own inverse depth is unavailable here, so use a
    # smooth map; the test checks the wiring (inference once, loss every step, log line), not the prior.
    calls = []

    def fake_predict(images, device):
        calls.append(tuple(images.shape))
        n, H, W = images.shape[:3]
        return torch.linspace(1, 2, H)[None, :, None].expand(n, H, W).to(torch.float16).contiguous()

    monkeypatch.setattr(tr, "predict_disparity", fake_predict)
    p = dataclasses.replace(PROFILES["laptop"], iterations=300, mono_lambda=0.1, mono_patch=16)
    with caplog.at_level(logging.INFO, logger="recon3d.recon.trainer"):
        splats = train(synthetic_ts[0], p, on_progress=lambda p: None)
    assert splats.means.shape[0] > 0
    assert len(calls) == 1 and calls[0][0] == synthetic_ts[0].images.shape[0]
    msgs = [r.getMessage() for r in caplog.records if "单目深度约束" in r.getMessage()]
    assert len(msgs) == 1 and "/300 步" in msgs[0]
    assert not msgs[0].startswith("单目深度约束：0/")


@pytest.mark.gpu
def test_trains_with_affine_colour_correction(synthetic_ts, caplog):
    p = dataclasses.replace(PROFILES["laptop"], iterations=300, appearance_lr=1e-3)
    with caplog.at_level(logging.INFO, logger="recon3d.recon.trainer"):
        splats = train(synthetic_ts[0], p, on_progress=lambda p: None)
    assert splats.means.shape[0] > 0
    msgs = [r.getMessage() for r in caplog.records if "逐帧颜色校正" in r.getMessage()]
    assert len(msgs) == 1
    # The correction moved off the identity (non-zero gains / offsets reported).
    assert "最大 0.000" not in msgs[0]


def test_contrib_pass_skipped_soon_after_opacity_reset():
    from recon3d.recon.trainer import _contrib_pass_allowed

    assert _contrib_pass_allowed(200, None)  # no reset yet
    assert not _contrib_pass_allowed(300, 200)
    assert not _contrib_pass_allowed(699, 200)
    assert _contrib_pass_allowed(700, 200)  # gap reached
    assert _contrib_pass_allowed(3000, 3000)  # same step: the reset runs after the pass
    assert _contrib_pass_allowed(200, 100, min_gap=100) and not _contrib_pass_allowed(199, 100, min_gap=100)
    # shipped profiles: laptop resets at 3000/6000 (refine_stop 7500), server up to 12000 (refine_stop 15000)
    for iters, last in ((15000, 6000), (30000, 12000)):
        assert all(_contrib_pass_allowed(s, last) for s in (iters // 2, 3 * iters // 4, iters))


def test_unknown_strategy_rejected():
    from recon3d.recon.dataset import TrainingSet

    ts = TrainingSet(images=torch.zeros(1, 4, 4, 3), K=torch.eye(3)[None], c2w=torch.eye(4)[None], depths=[None])
    with pytest.raises(ValueError, match="未知的训练策略"):
        train(ts, dataclasses.replace(PROFILES["laptop"], strategy="foo"), on_progress=lambda p: None, device="cpu")
