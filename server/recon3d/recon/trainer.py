"""3D Gaussian Splatting trainer on gsplat.

Parameter init and learning rates follow gsplat ``examples/simple_trainer.py``.
CUDA OOM is deliberately not caught here; the caller decides how to degrade.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from recon3d.config import Profile
from recon3d.recon.dataset import TrainingSet
from recon3d.recon.init_points import init_points

SH_C0 = 0.28209479177387814
PROGRESS_EVERY = 100
SH_DEGREE_INTERVAL = 1000
SSIM_LAMBDA = 0.2
PACKED = True  # packed projection: lower memory; the strategy must be told
RESET_EVERY = 3000  # opacity reset period (upstream default)


@dataclass
class Splats:
    """Trained Gaussians (raw, pre-activation parameters).

    means [N,3]; scales [N,3] log; quats [N,4] wxyz (unnormalized); opacities [N] logit;
    sh0 [N,1,3]; shN [N,K,3] with K = (sh_degree+1)^2 - 1.
    """

    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    sh0: torch.Tensor
    shN: torch.Tensor

    @property
    def sh_degree(self) -> int:
        return int(round(math.sqrt(self.shN.shape[1] + 1))) - 1


def _rasterize(params, viewmats, Ks, W, H, sh_degree):
    from gsplat import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    return rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=sh_degree,
        packed=PACKED,
    )


@torch.no_grad()
def render(splats: Splats, ts: TrainingSet) -> torch.Tensor:
    """Render every camera of ts. Returns [N,H,W,3] in 0..1 on the splats' device."""
    dev = splats.means.device
    H, W = ts.images.shape[1:3]
    params = {
        "means": splats.means, "quats": splats.quats, "scales": splats.scales,
        "opacities": splats.opacities, "sh0": splats.sh0, "shN": splats.shN,
    }
    viewmats = torch.linalg.inv(ts.c2w.to(dev))
    Ks = ts.K.to(dev)
    out = []
    for i in range(len(viewmats)):
        img, _, _ = _rasterize(params, viewmats[i : i + 1], Ks[i : i + 1], W, H, splats.sh_degree)
        out.append(img[0].clamp(0.0, 1.0))
    return torch.stack(out)


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of [B,3,H,W] images in 0..1 (11x11 Gaussian window, 'same' padding)."""
    c1, c2 = 0.01**2, 0.03**2
    pad = window.shape[-1] // 2
    mu_a = F.conv2d(a, window, padding=pad, groups=3)
    mu_b = F.conv2d(b, window, padding=pad, groups=3)
    s_aa = F.conv2d(a * a, window, padding=pad, groups=3) - mu_a**2
    s_bb = F.conv2d(b * b, window, padding=pad, groups=3) - mu_b**2
    s_ab = F.conv2d(a * b, window, padding=pad, groups=3) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2)
    return (num / den).mean()


def _knn_mean_sq_dist(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean squared distance to the k nearest other points, [M] (CPU KD-tree; O(M log M) memory-light)."""
    from scipy.spatial import cKDTree

    k_eff = min(k, len(xyz) - 1)
    if k_eff < 1:
        return np.full(len(xyz), 1e-4, np.float32)
    d, _ = cKDTree(xyz).query(xyz, k=k_eff + 1, workers=-1)
    return (d[:, 1:] ** 2).mean(axis=1).astype(np.float32)


def _scene_scale(c2w: torch.Tensor) -> float:
    centers = c2w[:, :3, 3]
    return float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1


def _init_params(ts: TrainingSet, profile: Profile, device: str) -> torch.nn.ParameterDict:
    xyz, rgb = init_points(ts)
    if len(xyz) > profile.max_gaussians:
        idx = np.random.default_rng(0).choice(len(xyz), profile.max_gaussians, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    means = torch.from_numpy(xyz).to(device)
    rgb_t = torch.from_numpy(rgb).to(device)
    n = len(means)
    dist2 = torch.from_numpy(_knn_mean_sq_dist(xyz)).to(device).clamp_min(1e-7)
    scales = torch.log(torch.sqrt(dist2))[:, None].repeat(1, 3)
    quats = torch.rand(n, 4, device=device, generator=torch.Generator(device).manual_seed(0))
    opacities = torch.logit(torch.full((n,), 0.1, device=device))
    K = (profile.sh_degree + 1) ** 2 - 1
    sh0 = ((rgb_t - 0.5) / SH_C0)[:, None, :]
    shN = torch.zeros(n, K, 3, device=device)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        }
    )


@torch.no_grad()
def _enforce_cap(params, optimizers, state, max_gaussians: int) -> None:
    """Prune lowest-opacity Gaussians if a densify step overshot max_gaussians."""
    from gsplat.strategy.ops import remove

    n = len(params["means"])
    if n <= max_gaussians:
        return
    order = torch.argsort(params["opacities"].flatten())
    mask = torch.zeros(n, dtype=torch.bool, device=order.device)
    mask[order[: n - max_gaussians]] = True
    remove(params=params, optimizers=optimizers, state=state, mask=mask)


def train(
    ts: TrainingSet,
    profile: Profile,
    on_progress: Callable[[float], None],
    device: str = "cuda",
) -> Splats:
    """Optimize Gaussians on ts. Calls on_progress(i/iterations) every 100 steps (and at the end)."""
    from gsplat import DefaultStrategy
    from gsplat.strategy import ops

    iters = profile.iterations
    # The image stack stays on the CPU (a real capture is GBs of float32); one frame is
    # copied to the GPU per step. Not pinned: pinning multi-GB host buffers is unreliable
    # under WSL and the per-step copy is ~1 ms.
    images = ts.images.cpu()
    viewmats = torch.linalg.inv(ts.c2w.to(device))
    Ks = ts.K.to(device)
    n_img, H, W = images.shape[:3]
    scene_scale = _scene_scale(ts.c2w.to(device))

    params = _init_params(ts, profile, device)
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {
        k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
        for k, lr in lrs.items()
    }
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / iters)
    )
    strategy = DefaultStrategy(
        refine_start_iter=min(500, iters // 10),
        refine_stop_iter=iters // 2,
        reset_every=RESET_EVERY,
        refine_every=100,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    window = _gaussian_window(device=device)
    gen = torch.Generator().manual_seed(0)

    for step in range(iters):
        i = int(torch.randint(n_img, (1,), generator=gen))
        sh_deg = min(step // SH_DEGREE_INTERVAL, profile.sh_degree)
        img, _, info = _rasterize(params, viewmats[i : i + 1], Ks[i : i + 1], W, H, sh_deg)
        gt = images[i : i + 1].to(device, non_blocking=True)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (img - gt).abs().mean()
        ssim_v = ssim(img.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), window)
        loss = (1.0 - SSIM_LAMBDA) * l1 + SSIM_LAMBDA * (1.0 - ssim_v)
        loss.backward()
        # Same order as upstream simple_trainer: optimizer step first, then the
        # strategy (grow/prune) on the updated params.
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        # The strategy keeps running at the cap so pruning continues; growth past the
        # cap is trimmed (lowest opacity first) right after.
        strategy.step_post_backward(params, optimizers, state, step, info, packed=PACKED)
        _enforce_cap(params, optimizers, state, profile.max_gaussians)
        # gsplat 1.5.3 never resets opacity (`step % reset_every == 0 & step > 0` is
        # always False by operator precedence), so do the strategy's reset here.
        if 0 < step < strategy.refine_stop_iter and step % strategy.reset_every == 0:
            ops.reset_opa(params=params, optimizers=optimizers, state=state, value=strategy.prune_opa * 2.0)
        if (step + 1) % PROGRESS_EVERY == 0 or step + 1 == iters:
            on_progress((step + 1) / iters)

    return Splats(**{k: v.detach() for k, v in params.items()})
