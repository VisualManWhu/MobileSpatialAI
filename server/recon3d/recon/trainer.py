"""3D Gaussian Splatting trainer on gsplat.

Parameter init and learning rates follow gsplat ``examples/simple_trainer.py``.
CUDA OOM is deliberately not caught here; the caller decides how to degrade.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recon3d.config import Profile
from recon3d.recon.bounds import SceneBounds
from recon3d.recon.dataset import TrainingSet
from recon3d.recon.init_points import init_points
from recon3d.recon.pose_opt import PoseRefine

log = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814
PROGRESS_EVERY = 100
SH_DEGREE_INTERVAL = 1000
SSIM_LAMBDA = 0.2
PACKED = True  # packed projection: lower memory; the strategy must be told
RESET_EVERY = 3000  # opacity reset period (upstream default)
POSE_REG = 1e-2  # weight of the per-camera pose-correction penalty
DEPTH_HUBER_DELTA = 0.1  # metres; ARCore depth noise is 5-20 cm


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


def _rasterize(params, viewmats, Ks, W, H, sh_degree, render_mode: str = "RGB"):
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
        render_mode=render_mode,
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


def depth_loss(
    rendered: torch.Tensor, target: torch.Tensor, conf: torch.Tensor, delta: float = DEPTH_HUBER_DELTA
) -> torch.Tensor | None:
    """Huber(|pooled(rendered) - target|, delta) weighted by conf, mean over conf>0.

    rendered [1,H,W] expected depth is average-pooled down to the depth map's [dh,dw].
    None if H % dh or W % dw != 0, or if no cell has conf > 0.
    """
    H, W = rendered.shape[-2:]
    dh, dw = target.shape
    if H % dh or W % dw:
        return None
    n = (conf > 0).sum()
    if n == 0:
        return None
    pooled = F.avg_pool2d(rendered[None], kernel_size=(H // dh, W // dw))[0, 0]
    h = F.huber_loss(pooled, target, reduction="none", delta=delta)
    return (conf * h).sum() / n


def _depth_view(ts: TrainingSet, i: int) -> torch.Tensor | None:
    """c2w of frame i's depth map when it differs from c2w[i] (raw depth paired with its own pose)."""
    dc2w = ts.depth_c2w[i] if i < len(ts.depth_c2w) else None
    if dc2w is None:
        return None
    if torch.allclose(dc2w.detach().cpu(), ts.c2w[i].detach().cpu(), atol=1e-6):
        return None
    return dc2w


def _depth_K(ts: TrainingSet, i: int, H: int, W: int) -> torch.Tensor:
    """[3,3] intrinsics of frame i's depth map: depth_K[i] or K[i] scaled to the depth size."""
    dK = ts.depth_K[i] if ts.depth_K is not None else None
    if dK is not None:
        return dK
    dh, dw = ts.depths[i].shape[-2:]
    S = torch.tensor([[dw / W, 0.0, 0.0], [0.0, dh / H, 0.0], [0.0, 0.0, 1.0]],
                     dtype=ts.K.dtype, device=ts.K.device)
    return S @ ts.K[i]


def _depth_loss_usable(ts: TrainingSet, i: int, H: int, W: int) -> bool:
    """Whether frame i's depth map can be compared with a depth render.

    With its own depth pose (depth_c2w[i] differs from c2w[i]) the depth is rendered
    directly at the depth map's size with the depth intrinsics, so only depth + confidence
    are needed. Otherwise the colour render is average-pooled to the depth grid, which
    assumes the depth map spans exactly the image field of view: no depth_K given, or
    depth_K[i] equals K[i] scaled by (dw/W, dh/H) (focal lengths within 1 % relative,
    principal point within 0.5 depth pixels), and the sizes must divide evenly.
    """
    depth = ts.depths[i]
    conf = ts.confs[i] if i < len(ts.confs) else None
    if depth is None or conf is None:
        return False
    if _depth_view(ts, i) is not None:
        return True
    dh, dw = depth.shape[-2:]
    if H % dh or W % dw:
        return False
    dK = ts.depth_K[i] if ts.depth_K is not None else None
    if dK is None:
        return True
    dK = dK.detach().cpu().double()
    K = ts.K[i].detach().cpu().double()
    sx, sy = dw / W, dh / H
    fx, fy, cx, cy = float(K[0, 0]) * sx, float(K[1, 1]) * sy, float(K[0, 2]) * sx, float(K[1, 2]) * sy
    return (
        abs(float(dK[0, 0]) - fx) <= 0.01 * abs(fx)
        and abs(float(dK[1, 1]) - fy) <= 0.01 * abs(fy)
        and abs(float(dK[0, 2]) - cx) <= 0.5
        and abs(float(dK[1, 2]) - cy) <= 0.5
    )


def reg_loss(params, opacity_reg: float, scale_reg: float) -> torch.Tensor:
    """Opacity and scale regularisers: opacity_reg * mean(sigmoid(opacity logits)) + scale_reg * mean(exp(log scales))."""
    total = torch.zeros((), device=params["opacities"].device)
    if opacity_reg:
        total = total + opacity_reg * torch.sigmoid(params["opacities"]).mean()
    if scale_reg:
        total = total + scale_reg * torch.exp(params["scales"]).mean()
    return total


def _log_pose_stats(pose: PoseRefine | None) -> None:
    if pose is None:
        return
    d = pose.delta.detach()
    rot = torch.rad2deg(d[:, :3].norm(dim=1))
    trans = d[:, 3:].norm(dim=1) * 100.0
    log.info(
        "位姿修正：旋转中位数 %.3f° / 最大 %.3f°，平移中位数 %.2f cm / 最大 %.2f cm",
        float(rot.median()), float(rot.max()), float(trans.median()), float(trans.max()),
    )


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


def _init_params(
    ts: TrainingSet, profile: Profile, device: str, bounds: SceneBounds | None = None
) -> torch.nn.ParameterDict:
    xyz, rgb = init_points(ts)
    if bounds is not None and bounds.enabled:
        keep = bounds.keep_np(xyz)
        if 0 < keep.sum() < len(xyz):
            log.info("初始点云裁掉场景范围外的 %d/%d 个点", len(xyz) - int(keep.sum()), len(xyz))
            xyz, rgb = xyz[keep], rgb[keep]
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


@torch.no_grad()
def _cull_outside(params, optimizers, state, bounds: SceneBounds | None) -> int:
    """Remove Gaussians outside the scene bounds; returns how many went. Never empties the set."""
    from gsplat.strategy.ops import remove

    if bounds is None or not bounds.enabled:
        return 0
    keep = bounds.keep(params["means"])
    n_out = int((~keep).sum())
    if n_out == 0 or n_out == len(keep):
        return 0
    remove(params=params, optimizers=optimizers, state=state, mask=~keep)
    return n_out


def _slice(params, mask: torch.Tensor) -> Splats:
    """The masked Gaussians as a detached Splats (on the params' device)."""
    return Splats(**{k: params[k][mask].detach().clone() for k in Splats.__dataclass_fields__})


def _cat_splats(parts: list[Splats]) -> Splats:
    return Splats(**{k: torch.cat([getattr(p, k) for p in parts]) for k in Splats.__dataclass_fields__})


@torch.no_grad()
def _prune_contrib(
    params, optimizers, state, viewmats, Ks, W: int, H: int, profile: Profile, relocatable: bool = False
) -> tuple[int, Splats | None]:
    """Remove Gaussians seen in too few views or contributing too little (recon/contrib.py).

    Returns (n_removed, removed Gaussians or None). Never empties the set. With relocatable (MCMC
    while relocation still runs) the marked Gaussians are not removed but get opacity 1e-4 so MCMC
    relocates them; the returned slice is taken before that, so it shows what was pruned.
    """
    from gsplat.strategy.ops import remove

    from recon3d.recon.contrib import compute_contrib, prune_mask

    total, nviews = compute_contrib(params, viewmats, Ks, W, H, downscale=2)
    mask = prune_mask(total, nviews, profile.contrib_min_views, profile.contrib_min_px)
    n = int(mask.sum())
    if n == 0 or n == len(mask):
        return 0, None
    removed = _slice(params, mask)
    if relocatable:
        params["opacities"].data[mask] = torch.logit(torch.tensor(1e-4)).item()
    else:
        remove(params=params, optimizers=optimizers, state=state, mask=mask)
    return n, removed


@torch.no_grad()
def _prune_sor(params, optimizers, state, sigma: float) -> tuple[int, Splats | None]:
    """Remove statistical outliers among the Gaussian centres (k = 8). Never empties the set."""
    from gsplat.strategy.ops import remove

    from recon3d.recon.outliers import sor_mask

    out = sor_mask(params["means"].detach().cpu().numpy(), k=8, sigma=sigma)
    n = int(out.sum())
    if n == 0 or n == len(out):
        return 0, None
    mask = torch.from_numpy(out).to(params["means"].device)
    removed = _slice(params, mask)
    remove(params=params, optimizers=optimizers, state=state, mask=mask)
    return n, removed


def train(
    ts: TrainingSet,
    profile: Profile,
    on_progress: Callable[[float], None],
    device: str = "cuda",
    pruned_out: Path | None = None,
) -> Splats:
    """Optimize Gaussians on ts. Calls on_progress(i/iterations) every 100 steps (and at the end).

    With pruned_out, every Gaussian removed by contribution pruning or the outlier filter
    (not bounds culling) is written there as a PLY once training ends (nothing written if none).
    """
    if profile.strategy not in ("default", "mcmc"):
        raise ValueError(f"未知的训练策略：{profile.strategy}")
    from gsplat import DefaultStrategy, MCMCStrategy
    from gsplat.strategy import ops

    mcmc = profile.strategy == "mcmc"
    iters = profile.iterations
    # The image stack stays on the CPU (a real capture is GBs of float32); one frame is
    # copied to the GPU per step. Not pinned: pinning multi-GB host buffers is unreliable
    # under WSL and the per-step copy is ~1 ms.
    images = ts.images.cpu()
    viewmats = torch.linalg.inv(ts.c2w.to(device))
    Ks = ts.K.to(device)
    n_img, H, W = images.shape[:3]
    scene_scale = _scene_scale(ts.c2w.to(device))
    bounds = SceneBounds.from_c2w(ts.c2w, profile.far_m, profile.bbox_margin_m)
    culled = 0
    use_contrib = profile.contrib_min_views > 0 or profile.contrib_min_px > 0
    contrib_steps = {iters // 2, 3 * iters // 4} if use_contrib else set()
    pruned: list[Splats] = []

    def _contrib(at_step: int) -> None:
        # Renders use the uncorrected viewmats (PoseRefine deltas ignored; pose_lr is 0 by default).
        # Lowering opacity only helps while MCMC still relocates; later passes remove.
        relocatable = mcmc and at_step < strategy.refine_stop_iter
        n, removed = _prune_contrib(params, optimizers, rm_state, viewmats, Ks, W, H, profile, relocatable)
        if removed is not None and pruned_out is not None:
            pruned.append(removed)
        log.info(
            "贡献度剪枝（步 %d）：移除 %d 个高斯（<%d 个视角可见或贡献<%.1f px），剩余 %d" + ("（MCMC：压低不透明度）" if relocatable else ""),
            at_step, n, profile.contrib_min_views, profile.contrib_min_px, len(params["means"]),
        )

    params = _init_params(ts, profile, device, bounds)
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
    if mcmc:
        # cap_max bounds the count, so _enforce_cap is not needed; MCMC has no opacity reset.
        strategy = MCMCStrategy(
            cap_max=profile.max_gaussians,
            refine_start_iter=min(500, iters // 10),
            refine_stop_iter=iters // 2,
            refine_every=100,
            min_opacity=0.005,
        )
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state()
        # ops.remove slices every tensor in state, but MCMC's only entry ("binoms") is a lookup
        # table, not per-Gaussian; MCMC keeps no per-Gaussian running state, so removals get none.
        rm_state: dict = {}
    else:
        strategy = DefaultStrategy(
            refine_start_iter=min(500, iters // 10),
            refine_stop_iter=iters // 2,
            reset_every=RESET_EVERY,
            refine_every=100,
        )
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=scene_scale)
        rm_state = state
    window = _gaussian_window(device=device)
    gen = torch.Generator().manual_seed(0)
    t_start = time.monotonic()
    use_depth = profile.depth_lambda > 0
    # Pose refinement starts with densification, once the initial cloud has settled.
    pose: PoseRefine | None = None
    pose_opt = None
    # Depth usability is decided once per frame; per-step cost is a list lookup.
    has_depth = [ts.depths[i] is not None for i in range(n_img)]
    depth_ok = [use_depth and _depth_loss_usable(ts, i, H, W) for i in range(n_img)]
    # Frames whose depth was captured from its own pose get a second, depth-only render
    # from that pose at the depth map's size (viewmat + K per frame, None = pooled path).
    depth_vms = [None] * n_img
    depth_Ks = [None] * n_img
    for i in range(n_img):
        if not depth_ok[i]:
            continue
        dv = _depth_view(ts, i)
        if dv is not None:
            depth_vms[i] = torch.linalg.inv(dv.to(device))[None]
            depth_Ks[i] = _depth_K(ts, i, H, W).to(device)[None]
    n_depth_view = sum(v is not None for v in depth_vms)
    depth_used = [False] * n_img
    warned_skip = False

    def _warn_skip(i: int, reason: str) -> None:
        nonlocal warned_skip
        if not warned_skip:
            log.warning("深度约束跳过：帧 %d %s（之后同类情况不再提示）", i, reason)
            warned_skip = True

    if use_depth:
        for i in range(n_img):
            if has_depth[i] and not depth_ok[i]:
                _warn_skip(i, "的深度图与图像不对齐或尺寸不整除")
                break

    for step in range(iters):
        if pose is None and profile.pose_lr > 0 and step >= strategy.refine_start_iter:
            pose = PoseRefine(n_img, device)
            pose_opt = torch.optim.Adam(pose.parameters(), lr=profile.pose_lr)
        i = int(torch.randint(n_img, (1,), generator=gen))
        sh_deg = min(step // SH_DEGREE_INTERVAL, profile.sh_degree)
        vm = viewmats[i : i + 1]
        idx = torch.tensor([i], device=device)
        if pose is not None:
            vm = pose(vm, idx)
        render_mode = "RGB+ED" if depth_ok[i] and depth_vms[i] is None else "RGB"
        out, _, info = _rasterize(params, vm, Ks[i : i + 1], W, H, sh_deg, render_mode)
        img = out[..., :3]
        gt = images[i : i + 1].to(device, non_blocking=True)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (img - gt).abs().mean()
        ssim_v = ssim(img.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), window)
        loss = (1.0 - SSIM_LAMBDA) * l1 + SSIM_LAMBDA * (1.0 - ssim_v)
        if depth_ok[i]:
            if depth_vms[i] is not None:
                dvm = depth_vms[i]
                if pose is not None:
                    # The per-camera correction is a trajectory error; apply it to the depth view too.
                    dvm = pose(dvm, idx)
                dh, dw = ts.depths[i].shape[-2:]
                pred = _rasterize(params, dvm, depth_Ks[i], dw, dh, 0, "ED")[0][..., 0]
            else:
                pred = out[..., 3]
            dl = depth_loss(pred, ts.depths[i].to(device), ts.confs[i].to(device))
            if dl is not None:
                loss = loss + profile.depth_lambda * dl
                depth_used[i] = True
            else:
                _warn_skip(i, "没有可用的深度像素（置信度全为 0）")
        if pose is not None:
            loss = loss + POSE_REG * pose.reg()
        if profile.opacity_reg or profile.scale_reg:
            loss = loss + reg_loss(params, profile.opacity_reg, profile.scale_reg)
        loss.backward()
        if pose_opt is not None:
            pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
        # Same order as upstream simple_trainer: optimizer step first, then the
        # strategy (grow/prune) on the updated params.
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        # The strategy keeps running at the cap so pruning continues; growth past the
        # cap is trimmed (lowest opacity first) right after.
        if mcmc:
            strategy.step_post_backward(params, optimizers, state, step, info, lr=means_sched.get_last_lr()[0])
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=PACKED)
        # Floaters that densified or drifted out of the room go at the strategy's cadence.
        if step > 0 and step % strategy.refine_every == 0:
            culled += _cull_outside(params, optimizers, rm_state, bounds)
        if step in contrib_steps:
            _contrib(step)
        if not mcmc:
            _enforce_cap(params, optimizers, state, profile.max_gaussians)
            # gsplat 1.5.3 never resets opacity (`step % reset_every == 0 & step > 0` is
            # always False by operator precedence), so do the strategy's reset here.
            if 0 < step < strategy.refine_stop_iter and step % strategy.reset_every == 0:
                ops.reset_opa(params=params, optimizers=optimizers, state=state, value=strategy.prune_opa * 2.0)
        if (step + 1) % PROGRESS_EVERY == 0 or step + 1 == iters:
            on_progress((step + 1) / iters)

    if use_depth:
        log.info(
            "深度约束用于 %d/%d 个有深度的帧（其中 %d 帧按深度自身位姿单独渲染）",
            sum(depth_used), sum(has_depth), n_depth_view,
        )
    if use_contrib:
        _contrib(iters)
    culled += _cull_outside(params, optimizers, rm_state, bounds)
    if profile.sor_sigma > 0:
        n, removed = _prune_sor(params, optimizers, rm_state, profile.sor_sigma)
        if removed is not None and pruned_out is not None:
            pruned.append(removed)
        log.info("离群点滤波（k=8，%.1fσ）：移除 %d 个高斯，剩余 %d", profile.sor_sigma, n, len(params["means"]))
    if pruned_out is not None and pruned:
        from recon3d.recon.export_splat import export_ply

        export_ply(_cat_splats(pruned), pruned_out)
        log.info("被剪高斯已写入 %s（%d 个）", pruned_out, sum(len(p.means) for p in pruned))
    if bounds.enabled:
        log.info(
            "场景范围裁剪（距最近相机 ≤ %.1f m，相机包围盒外扩 %.1f m）：训练中共移除 %d 个高斯，剩余 %d",
            profile.far_m, profile.bbox_margin_m, culled, len(params["means"]),
        )
    log.info("训练耗时 %.1f s（%d 步）", time.monotonic() - t_start, iters)
    _log_pose_stats(pose)
    return Splats(**{k: v.detach() for k, v in params.items()})
