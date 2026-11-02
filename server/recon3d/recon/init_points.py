"""Initial point cloud from back-projected ARCore depth."""
from __future__ import annotations

import numpy as np

from recon3d.recon.dataset import TrainingSet

# Minimum per-pixel confidence (0..1) for a depth sample to seed a Gaussian.
# confs already encode the valid range (0.3..MAX_DEPTH_M) and the depth-edge penalty.
CONF_MIN = 0.5


def _voxel_mean(xyz: np.ndarray, rgb: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    n = len(counts)
    sx = np.zeros((n, 3), np.float64)
    sc = np.zeros((n, 3), np.float64)
    np.add.at(sx, inv, xyz)
    np.add.at(sc, inv, rgb)
    return (sx / counts[:, None]).astype(np.float32), (sc / counts[:, None]).astype(np.float32)


def init_points(
    ts: TrainingSet,
    voxel_size: float = 0.02,
    min_points: int = 5000,
    seed: int = 0,
    conf_min: float = CONF_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depths (meters), keeping pixels with confidence >= conf_min to world space.

    Returns (xyz [M,3], rgb [M,3] in 0..1), float32. If fewer than min_points
    survive, pads with seeded uniform random gray points in the camera bbox + 1 m.
    """
    images = ts.images.detach().cpu().numpy()
    Ks = ts.K.detach().cpu().numpy().astype(np.float64)
    c2ws = ts.c2w.detach().cpu().numpy().astype(np.float64)
    H, W = images.shape[1:3]

    xyz_all, rgb_all = [], []
    for i, depth in enumerate(ts.depths):
        conf = ts.confs[i]
        if depth is None or conf is None:
            continue
        d = depth.detach().cpu().numpy().astype(np.float64)
        dh, dw = d.shape
        c = conf.detach().cpu().numpy()
        valid = (c > 0) & (c >= conf_min) & (d > 0) & np.isfinite(d)
        v, u = np.nonzero(valid)
        if len(v) == 0:
            continue
        z = d[v, u]
        dK = ts.depth_K[i] if ts.depth_K is not None else None
        if dK is not None:
            dK = dK.detach().cpu().numpy().astype(np.float64)
            fx, fy, cx, cy = dK[0, 0], dK[1, 1], dK[0, 2], dK[1, 2]
        else:
            K = Ks[i]
            fx, fy = K[0, 0] * dw / W, K[1, 1] * dh / H
            cx, cy = K[0, 2] * dw / W, K[1, 2] * dh / H
        # pixel-center convention is the same as ARCore intrinsics: use integer pixel coords
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        pts_c = np.stack([x, y, z, np.ones_like(z)], axis=1)
        pts_w = (c2ws[i] @ pts_c.T).T[:, :3]
        iu = np.clip(np.round(u * W / dw).astype(int), 0, W - 1)
        iv = np.clip(np.round(v * H / dh).astype(int), 0, H - 1)
        xyz_all.append(pts_w)
        rgb_all.append(images[i][iv, iu].astype(np.float64))

    if xyz_all:
        xyz, rgb = _voxel_mean(np.concatenate(xyz_all), np.concatenate(rgb_all), voxel_size)
    else:
        xyz, rgb = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    if len(xyz) < min_points:
        cams = c2ws[:, :3, 3]
        lo, hi = cams.min(axis=0) - 1.0, cams.max(axis=0) + 1.0
        rng = np.random.default_rng(seed)
        n = min_points - len(xyz)
        extra = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
        xyz = np.concatenate([xyz, extra]).astype(np.float32)
        rgb = np.concatenate([rgb, np.full((n, 3), 0.5, np.float32)]).astype(np.float32)
    return xyz, rgb
