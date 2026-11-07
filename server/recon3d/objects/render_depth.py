"""Depth rendered from the trained Gaussians, and the Gaussian centres in scene.splat order.

Captured ARCore depth is low-resolution, has holes on screens / glass / dark surfaces and (raw) lags
the colour frame; the trained scene has none of those problems and is, by construction, consistent
with the Gaussians we assign to objects (spec 2026-11-07 §4.4 "深度来源")."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from recon3d.recon.export_splat import SPLAT_DTYPE

ALPHA_MIN = 0.5  # pixels thinner than this are "no surface" (depth 0)


def read_splat_positions(path: Path) -> np.ndarray:
    """[N, 3] float32 Gaussian centres in the order the .splat stores them (= object id order)."""
    raw = np.frombuffer(Path(path).read_bytes(), dtype=SPLAT_DTYPE)
    return np.ascontiguousarray(raw["pos"], dtype=np.float32)


def render_depths(
    ply_path: Path,
    c2w_cv: np.ndarray,
    Ks: np.ndarray,
    width: int,
    height: int,
    device: str = "cuda",
    scale: float = 0.5,
) -> Iterator[np.ndarray]:
    """Yields one expected-depth map [h, w] float32 (metres, 0 = no surface) per camera, rendered at
    `scale` of the image size with intrinsics scaled to match. Frees the Gaussians when exhausted."""
    import torch

    from recon3d.recon.export_splat import load_ply
    from recon3d.recon.trainer import _rasterize

    h, w = round(height * scale), round(width * scale)
    splats = load_ply(ply_path, device=device)
    params = {k: getattr(splats, k) for k in ("means", "scales", "quats", "opacities", "sh0", "shN")}
    try:
        with torch.no_grad():
            for c2w, K in zip(np.asarray(c2w_cv, dtype=np.float64), np.asarray(Ks, dtype=np.float64)):
                viewmat = torch.from_numpy(np.linalg.inv(c2w)).float().to(device)[None]
                Ks_t = torch.from_numpy(K).float().to(device)[None]
                Ks_t[:, 0, :] *= w / width
                Ks_t[:, 1, :] *= h / height
                out, alphas, _ = _rasterize(params, viewmat, Ks_t, w, h, splats.sh_degree, render_mode="ED")
                depth = out[0, ..., 0]
                depth = torch.where(alphas[0, ..., 0] >= ALPHA_MIN, depth, torch.zeros_like(depth))
                yield depth.float().cpu().numpy()
    finally:
        del params, splats
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
