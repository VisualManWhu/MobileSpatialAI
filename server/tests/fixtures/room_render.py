"""CPU ray-cast renderer of a textured box room, for pycolmap tests.

The room is the inside of the axis-aligned box [-3,3] x [-1.5,1.5] x [-3,3] m
(x, y-up, z). Each of the six faces carries a blobby random texture so SIFT
finds repeatable features. Poses are OpenCV camera-to-world (x right, y down,
z forward).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

HALF = np.array([3.0, 1.5, 3.0])
TEX = 256


def _texture(seed: int, face: int) -> np.ndarray:
    rng = np.random.default_rng(seed + face)
    tex = rng.integers(0, 256, (TEX, TEX, 3)).astype(np.float64)
    tex = gaussian_filter(tex, (1.5, 1.5, 0))
    lo, hi = tex.min(), tex.max()
    return (tex - lo) / max(hi - lo, 1e-9) * 255.0


def render_room(c2w_cv: np.ndarray, K: np.ndarray, W: int, H: int, seed: int = 0) -> np.ndarray:
    """Render the room from a pinhole camera; returns uint8 H x W x 3."""
    c2w = np.asarray(c2w_cv, dtype=np.float64)
    R, o = c2w[:3, :3], c2w[:3, 3]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    pix = np.stack([u.ravel(), v.ravel(), np.ones(W * H)])          # [3, P]
    d = R @ (np.linalg.inv(K) @ pix)                                 # [3, P]
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.where(d > 0, HALF[:, None], -HALF[:, None])
        t = (bound - o[:, None]) / d                                 # [3, P]
    t = np.where(np.isfinite(t) & (t > 0), t, np.inf)
    axis = np.argmin(t, axis=0)                                      # [P]
    cols = np.arange(t.shape[1])
    t_hit = t[axis, cols]
    p = o[:, None] + t_hit[None, :] * d                              # [3, P]
    face = axis * 2 + (d[axis, cols] > 0).astype(int)
    out = np.zeros((W * H, 3))
    for f in range(6):
        sel = np.nonzero(face == f)[0]
        if sel.size == 0:
            continue
        a = f // 2
        others = [i for i in range(3) if i != a]
        s0 = (p[others[0], sel] + HALF[others[0]]) / (2 * HALF[others[0]])
        s1 = (p[others[1], sel] + HALF[others[1]]) / (2 * HALF[others[1]])
        coords = np.stack([np.clip(s0, 0, 1) * (TEX - 1), np.clip(s1, 0, 1) * (TEX - 1)])
        tex = _texture(seed, f)
        for c in range(3):
            out[sel, c] = map_coordinates(tex[:, :, c], coords, order=1, mode="nearest")
    return np.clip(out, 0, 255).astype(np.uint8).reshape(H, W, 3)


def _look_at_cv(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    dn = np.cross(f, r)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, dn, f, eye
    return c2w


def loop_trajectory(n: int, radius: float = 1.2, bob: float = 0.25) -> np.ndarray:
    """n OpenCV c2w poses on a circle, each looking outward (horizontally) at the walls.

    The height bobs by +-bob m around y = 0: COLMAP's Sim3 alignment of the
    reconstruction onto the position priors rejects exactly coplanar centres.
    """
    up = np.array([0.0, 1.0, 0.0])
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        radial = np.array([np.sin(a), 0.0, np.cos(a)])
        eye = radius * radial + np.array([0.0, bob * np.sin(2 * a), 0.0])
        out.append(_look_at_cv(eye, eye + radial, up))
    return np.stack(out)
