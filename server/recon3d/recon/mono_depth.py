"""Monocular depth prior: Depth Anything V2 (Small) relative disparity + patch-wise Pearson loss.

ARCore depth is motion stereo: on white walls and floors there is no parallax to triangulate, so it is
least reliable exactly where floaters form (which is why the ARCore depth loss did nothing, see
docs/acceptance/phase1.md section 5). A monocular network relies on semantic priors and is steadiest on
planar, textureless regions, so it is genuinely new information. Its output is relative (affine-invariant
inverse depth), hence the loss compares the *rendered* inverse depth with it through the Pearson
correlation inside 64x64 patches, which is invariant to a per-patch scale and offset.

Inference runs once before training on every frame (a few seconds for 300 frames on a 4060) and the
model is released before the Gaussians are allocated, so it never competes with training for memory.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
# DINOv2 backbone: inputs must be multiples of the 14 px patch; the short side 518 is the model's native size.
PATCH_PX = 14
SHORT_SIDE = 518
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _load_model(device: str):
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    if device.startswith("cuda"):
        model = model.half()

    def run(x: torch.Tensor) -> torch.Tensor:
        return model(pixel_values=x.to(next(model.parameters()).dtype)).predicted_depth

    return run


def _input_size(H: int, W: int) -> tuple[int, int]:
    """(h, w) with short side SHORT_SIDE, both multiples of PATCH_PX, aspect kept."""
    s = SHORT_SIDE / min(H, W)
    h = max(PATCH_PX, round(H * s / PATCH_PX) * PATCH_PX)
    w = max(PATCH_PX, round(W * s / PATCH_PX) * PATCH_PX)
    return h, w


@torch.no_grad()
def predict_disparity(
    images: torch.Tensor,
    device: str,
    batch: int = 4,
    model: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Relative disparity [N,H,W] float16 (CPU) for images [N,H,W,3] in [0,1] (any device).

    Larger = closer; scale and offset are arbitrary per frame. `model` maps a normalised batch
    [B,3,h,w] to disparity [B,h',w'] (resized back here); None loads Depth Anything V2 Small.
    """
    n, H, W = images.shape[:3]
    h, w = _input_size(H, W)
    t0 = time.monotonic()
    run = model if model is not None else _load_model(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out = torch.empty(n, H, W, dtype=torch.float16)
    for s in range(0, n, batch):
        x = images[s : s + batch].to(device).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        d = run(x).float()
        d = F.interpolate(d[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        out[s : s + batch] = d.to(torch.float16).cpu()
    del run
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    log.info("单目深度（Depth Anything V2 Small）：%d 帧 %dx%d，用时 %.1f s", n, W, H, time.monotonic() - t0)
    return out


def pearson_patch_loss(
    depth: torch.Tensor,
    disparity: torch.Tensor,
    mask: torch.Tensor,
    patch: int = 64,
    min_fill: float = 0.1,
) -> torch.Tensor | None:
    """mean over patches of (1 - Pearson(1/depth, disparity)) using only masked pixels.

    depth, disparity, mask: [H,W] (rendered depth in metres, monocular disparity, bool/0-1 mask of
    pixels to use, e.g. rendered alpha > 0.5). The image is cut into patch x patch tiles (a partial
    border is dropped); a tile takes part when at least min_fill of its pixels are masked. Returns None
    when no tile qualifies.
    """
    H, W = depth.shape
    ph, pw = H // patch, W // patch
    if ph == 0 or pw == 0:
        return None

    def tiles(x: torch.Tensor) -> torch.Tensor:
        return x[: ph * patch, : pw * patch].reshape(ph, patch, pw, patch).permute(0, 2, 1, 3).reshape(-1, patch * patch)

    m = tiles(mask.to(depth.dtype))
    a = tiles(1.0 / depth.clamp(min=1e-3))
    b = tiles(disparity.to(depth.dtype))
    n = m.sum(dim=1)
    valid = n >= min_fill * patch * patch
    if not valid.any():
        return None
    m, a, b, n = m[valid], a[valid], b[valid], n[valid, None]
    ma = (a * m).sum(dim=1, keepdim=True) / n
    mb = (b * m).sum(dim=1, keepdim=True) / n
    da = (a - ma) * m
    db = (b - mb) * m
    cov = (da * db).sum(dim=1)
    va = (da * da).sum(dim=1)
    vb = (db * db).sum(dim=1)
    corr = cov / (va.sqrt() * vb.sqrt() + 1e-6)
    return (1.0 - corr).mean()
