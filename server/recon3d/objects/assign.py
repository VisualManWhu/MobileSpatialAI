"""Gaussian -> object assignment by projecting Gaussian centres into every sampled frame and voting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ObjectMasks:
    frame: int  # index into the sampled frame list (matches depths / c2w / Ks)
    object_id: int  # 1-based object id
    mask: np.ndarray  # [h, w] bool at depth resolution


def assign_gaussians(
    positions: np.ndarray,
    c2w_cv: np.ndarray,
    Ks_depth: np.ndarray,
    depths: list[np.ndarray],
    masks: list[ObjectMasks],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    device: str = "cpu",
    min_votes: int = 2,
    depth_tol_m: float = 0.1,
    depth_tol_rel: float = 0.05,
    bbox_margin_m: float = 0.1,
    chunk: int = 200_000,
) -> np.ndarray:
    """Per-Gaussian object id (uint16, 0 = none). A Gaussian votes for object o in frame f when its
    centre projects inside o's mask in f and its depth agrees with the rendered depth there. The
    object with most votes wins if votes >= min_votes and the centre lies within that object's
    bbox grown by bbox_margin_m. Ties go to the lower id."""
    n = len(positions)
    n_obj = len(bbox_min)
    out = np.zeros(n, dtype=np.uint16)
    if n == 0 or n_obj == 0 or not masks:
        return out

    # One label image per frame: 0 = no object; later masks of the same frame overwrite earlier ones
    # only where they are set, so overlapping masks keep the first object (ids are merge order).
    labels: dict[int, torch.Tensor] = {}
    for m in masks:
        lab = labels.get(m.frame)
        if lab is None:
            lab = torch.zeros(m.mask.shape, dtype=torch.int32)
            labels[m.frame] = lab
        lab[(torch.from_numpy(m.mask) & (lab == 0))] = int(m.object_id)

    pos_all = torch.from_numpy(np.asarray(positions, dtype=np.float32))
    w2c = torch.from_numpy(np.linalg.inv(np.asarray(c2w_cv, dtype=np.float64))).float()
    Ks_t = torch.from_numpy(np.asarray(Ks_depth, dtype=np.float32))
    bmin = torch.from_numpy(np.asarray(bbox_min, dtype=np.float32)) - bbox_margin_m
    bmax = torch.from_numpy(np.asarray(bbox_max, dtype=np.float32)) + bbox_margin_m

    for start in range(0, n, chunk):
        p = pos_all[start : start + chunk].to(device)  # [c, 3]
        c = len(p)
        votes = torch.zeros((c, n_obj + 1), dtype=torch.int16, device=device)
        ph = torch.cat([p, torch.ones(c, 1, device=device)], dim=1)  # [c, 4]
        for f, lab_cpu in labels.items():
            depth = torch.from_numpy(np.asarray(depths[f], dtype=np.float32)).to(device)
            lab = lab_cpu.to(device)
            h, w = depth.shape
            cam = ph @ w2c[f].to(device).T  # [c, 4]
            z = cam[:, 2]
            ok = z > 1e-3
            K = Ks_t[f].to(device)
            u = torch.round(K[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + K[0, 2]).long()
            v = torch.round(K[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + K[1, 2]).long()
            ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
            uu = u.clamp(0, w - 1)
            vv = v.clamp(0, h - 1)
            d = depth[vv, uu]
            obj = lab[vv, uu].long()
            ok &= (d > 0) & ((z - d).abs() < depth_tol_m + depth_tol_rel * d) & (obj > 0)
            votes[torch.arange(c, device=device)[ok], obj[ok]] += 1
        votes[:, 0] = 0
        best = votes.argmax(dim=1)  # lowest id on ties
        nv = votes.gather(1, best[:, None])[:, 0]
        inside = ((p[:, None, :] >= bmin.to(device)[None]) & (p[:, None, :] <= bmax.to(device)[None])).all(dim=2)
        ok_in = inside.gather(1, (best - 1).clamp_min(0)[:, None])[:, 0]
        keep = (nv >= min_votes) & (best > 0) & ok_in
        res = torch.where(keep, best, torch.zeros_like(best))
        out[start : start + c] = res.to(torch.int64).cpu().numpy().astype(np.uint16)
    return out
