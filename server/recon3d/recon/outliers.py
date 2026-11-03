"""Classic statistical outlier removal on a point cloud (Gaussian centres)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def sor_mask(xyz: np.ndarray, k: int = 8, sigma: float = 3.0) -> np.ndarray:
    """Bool mask [M], True = outlier.

    A point is an outlier when its mean distance to its ``k`` nearest other points exceeds
    ``mean + sigma * std`` of that statistic over all points. All-False when ``M <= k`` or ``sigma <= 0``.
    """
    xyz = np.asarray(xyz)
    m = len(xyz)
    if m <= k or k < 1 or sigma <= 0:
        return np.zeros(m, dtype=bool)
    # k+1 neighbours: column 0 is the point itself.
    dist, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    mean_d = dist[:, 1:].mean(axis=1)
    return mean_d > mean_d.mean() + sigma * mean_d.std()
