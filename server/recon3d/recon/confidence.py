"""Per-pixel depth confidence: sensor confidence map or a depth-edge heuristic."""
from __future__ import annotations

import numpy as np

MAX_DEPTH_M = 4.0
EDGE_M = 0.10


def depth_confidence(depth_m: np.ndarray, conf_u8: np.ndarray | None) -> np.ndarray:
    """Return float32 HxW confidence in 0..1 for a depth map in meters.

    Invalid depth (<= 0, non-finite, > MAX_DEPTH_M) always gets 0. With conf_u8 the
    confidence is conf_u8 / 255; otherwise it falls off with the largest depth jump
    to a 4-neighbour: 1 - g / EDGE_M, clipped to 0..1.
    """
    d = np.asarray(depth_m, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(d) & (d > 0) & (d <= MAX_DEPTH_M)
    if conf_u8 is not None:
        return np.where(valid, np.asarray(conf_u8, np.float32) / 255.0, 0.0).astype(np.float32)

    with np.errstate(invalid="ignore"):
        ok = np.isfinite(d) & (d > 0)
    dz = np.where(ok, d, 0.0).astype(np.float32)
    dp = np.pad(dz, 1, mode="edge")
    okp = np.pad(ok, 1, mode="edge")
    g = np.zeros_like(dz)
    for sl in (
        (slice(0, -2), slice(1, -1)),
        (slice(2, None), slice(1, -1)),
        (slice(1, -1), slice(0, -2)),
        (slice(1, -1), slice(2, None)),
    ):
        nb = np.where(okp[sl], dp[sl], dz)  # invalid neighbour counts as the pixel itself
        g = np.maximum(g, np.abs(dz - nb))
    conf = np.clip(1.0 - g / EDGE_M, 0.0, 1.0)
    return np.where(valid, conf, 0.0).astype(np.float32)
