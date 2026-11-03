"""Helpers for the off-trajectory contact sheet (floaters hide behind geometry in training views)."""
from __future__ import annotations

import numpy as np
import torch


def raise_cameras(c2w: torch.Tensor, dy_m: float = 0.5) -> torch.Tensor:
    """Copy of [N,4,4] c2w with the world-Y translation increased by dy_m (ARCore world: +Y up)."""
    out = c2w.clone()
    out[:, 1, 3] += dy_m
    return out


def pick_views(n: int, k: int) -> list[int]:
    """min(n, k) evenly spaced distinct indices in [0, n)."""
    m = min(n, k)
    if m <= 0:
        return []
    if m == 1:
        return [0]
    return [int(i) for i in np.round(np.linspace(0, n - 1, m)).astype(int)]


def contact_sheet(imgs: list[np.ndarray], cols: int = 4) -> np.ndarray:
    """Tile equally sized uint8 HxWx3 images row-major; black tiles pad the last row."""
    h, w, c = imgs[0].shape
    rows = -(-len(imgs) // cols)
    sheet = np.zeros((rows * h, cols * w, c), np.uint8)
    for i, im in enumerate(imgs):
        r, col = divmod(i, cols)
        sheet[r * h : (r + 1) * h, col * w : (col + 1) * w] = im
    return sheet
