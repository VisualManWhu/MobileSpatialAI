"""Synthetic 3DGS scene: known colored Gaussians on a sphere, rendered by gsplat.

Shared by the trainer tests and (later) the synthetic-capture end-to-end test.
"""
from __future__ import annotations

import math

import torch

from recon3d.recon.dataset import TrainingSet

ORBIT_RADIUS = 4.0
SPHERE_RADIUS = 1.0
FOCAL_FRAC = 0.9  # fx = fy = FOCAL_FRAC * W


def look_at_c2w(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """OpenCV camera-to-world (x right, y down, z forward) looking from eye at target."""
    f = target - eye
    f = f / f.norm()
    r = torch.linalg.cross(f, up)
    r = r / r.norm()
    d = torch.linalg.cross(f, r)
    c2w = torch.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, d, f, eye
    return c2w


def orbit_c2ws(azimuths_deg, elevations_deg, radius: float = ORBIT_RADIUS) -> torch.Tensor:
    up = torch.tensor([0.0, 1.0, 0.0])  # world y-up
    out = []
    for az, el in zip(azimuths_deg, elevations_deg):
        a, e = math.radians(az), math.radians(el)
        eye = torch.tensor(
            [radius * math.cos(e) * math.sin(a), radius * math.sin(e), radius * math.cos(e) * math.cos(a)]
        )
        out.append(look_at_c2w(eye, torch.zeros(3), up))
    return torch.stack(out)


def make_gaussians(n: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """n Gaussians on a unit sphere with a smooth color pattern plus noise (activated values)."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    means = dirs * SPHERE_RADIUS
    colors = 0.5 + 0.4 * torch.sin(3.0 * means + torch.tensor([0.0, 2.0, 4.0]))
    colors = (colors + 0.08 * torch.randn(n, 3, generator=g)).clamp(0.0, 1.0)
    return {
        "means": means,
        "quats": torch.randn(n, 4, generator=g),
        "scales": torch.full((n, 3), 0.06) * (0.8 + 0.4 * torch.rand(n, 3, generator=g)),
        "opacities": torch.full((n,), 0.9),
        "colors": colors,
    }


def _render(gt: dict[str, torch.Tensor], c2w: torch.Tensor, K: torch.Tensor, W: int, H: int):
    from gsplat import rasterization

    dev = c2w.device
    out, alpha, _ = rasterization(
        gt["means"].to(dev),
        gt["quats"].to(dev),
        gt["scales"].to(dev),
        gt["opacities"].to(dev),
        gt["colors"].to(dev),
        torch.linalg.inv(c2w),
        K,
        W,
        H,
        render_mode="RGB+ED",
    )
    rgb = out[..., :3].clamp(0.0, 1.0)
    depth = torch.where(alpha[..., 0] > 0.5, out[..., 3], torch.zeros_like(out[..., 3]))
    return rgb, depth


def _build_ts(gt, c2ws: torch.Tensor, W: int, H: int, device: str) -> TrainingSet:
    c2ws = c2ws.to(device)
    f = FOCAL_FRAC * W
    K = torch.tensor([[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]], device=device)
    Ks = K.expand(len(c2ws), 3, 3).contiguous()
    with torch.no_grad():
        rgb, depth = _render(gt, c2ws, Ks, W, H)
    return TrainingSet(
        images=rgb.contiguous(), K=Ks, c2w=c2ws, depths=[d.contiguous() for d in depth]
    )


def make_synthetic_scene(
    n_gaussians: int = 2000,
    n_train: int = 24,
    n_test: int = 4,
    image_size: tuple[int, int] = (256, 192),
    seed: int = 0,
    device: str = "cuda",
) -> tuple[TrainingSet, TrainingSet, dict[str, torch.Tensor]]:
    """Render a known scene from orbit cameras.

    Train cameras: n_train evenly spaced azimuths, elevations alternating +-20 deg.
    Test cameras: n_test held-out azimuths (offset by half a train step), elevation 5 deg.
    Depth maps are expected depth (meters) at image resolution, 0 where alpha <= 0.5.
    Returns (train_ts, test_ts, gt) where gt holds the activated Gaussian parameters.
    """
    W, H = image_size
    gt = make_gaussians(n_gaussians, seed)
    step = 360.0 / n_train
    train_az = [i * step for i in range(n_train)]
    train_el = [20.0 if i % 2 == 0 else -20.0 for i in range(n_train)]
    test_az = [(i * 360.0 / n_test) + step / 2 for i in range(n_test)]
    test_el = [5.0] * n_test
    train_ts = _build_ts(gt, orbit_c2ws(train_az, train_el), W, H, device)
    test_ts = _build_ts(gt, orbit_c2ws(test_az, test_el), W, H, device)
    return train_ts, test_ts, gt
