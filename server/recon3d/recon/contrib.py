"""Per-Gaussian contribution and view count, used to prune floaters.

Trick: rasterize an all-ones per-Gaussian feature instead of colours. The rendered
image is sum_i w_i * 1 per pixel, so d(render.sum())/d(feat_i) is Gaussian i's
total blending weight (in pixels) in that view.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def compute_contrib(
    params: Mapping[str, torch.Tensor],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    W: int,
    H: int,
    downscale: int = 2,
    min_px: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (total [N] float32, nviews [N] int32).

    total: summed blending weight in pixels over all views at the downscaled size.
    nviews: number of views where the per-view weight exceeds min_px.
    """
    from gsplat import rasterization

    w, h = W // downscale, H // downscale
    sx, sy = w / W, h / H
    means = params["means"].detach()
    quats = params["quats"].detach()
    scales = torch.exp(params["scales"].detach())
    opacities = torch.sigmoid(params["opacities"].detach())
    n = means.shape[0]
    total = torch.zeros(n, dtype=torch.float32, device=means.device)
    nviews = torch.zeros(n, dtype=torch.int32, device=means.device)
    Ks = Ks.clone().float()
    if Ks.ndim == 2:
        Ks = Ks[None].expand(viewmats.shape[0], -1, -1).clone()
    Ks[:, 0, :] *= sx
    Ks[:, 1, :] *= sy
    with torch.enable_grad():
        for i in range(viewmats.shape[0]):
            feat = torch.ones(n, 1, device=means.device, requires_grad=True)
            render, _, _ = rasterization(
                means, quats, scales, opacities, feat,
                viewmats=viewmats[i : i + 1], Ks=Ks[i : i + 1],
                width=w, height=h, sh_degree=None, packed=False,
            )
            (g,) = torch.autograd.grad(render.sum(), feat)
            g = g[:, 0].float()
            total += g
            nviews += (g > min_px).to(torch.int32)
    return total, nviews


def prune_mask(total: torch.Tensor, nviews: torch.Tensor, min_views: int, min_px: float) -> torch.Tensor:
    """True = prune: seen by too few views or negligible total contribution."""
    return (nviews < min_views) | (total < min_px)
