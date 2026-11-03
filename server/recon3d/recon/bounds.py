"""Scene bounds derived from the camera path: everything far from every camera is noise.

Depth maps beyond a few metres are unreliable and Gaussians that densify or drift out of the room
render as floaters, so a Gaussian is kept only if it is
  * within ``far_m`` of the nearest camera position, and
  * inside the camera-position bounding box grown by ``bbox_margin_m`` on each side.
Either criterion is disabled with a value of 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

CHUNK = 65_536


@dataclass(frozen=True)
class SceneBounds:
    cams: torch.Tensor  # [N, 3] camera positions (world)
    far_m: float
    lo: torch.Tensor | None  # [3] grown bbox min, None = off
    hi: torch.Tensor | None  # [3]

    @classmethod
    def from_c2w(cls, c2w: torch.Tensor, far_m: float, bbox_margin_m: float) -> "SceneBounds":
        cams = c2w[:, :3, 3].detach().float()
        lo = hi = None
        if bbox_margin_m > 0:
            lo = cams.min(dim=0).values - bbox_margin_m
            hi = cams.max(dim=0).values + bbox_margin_m
        return cls(cams=cams, far_m=float(far_m), lo=lo, hi=hi)

    @property
    def enabled(self) -> bool:
        return self.far_m > 0 or self.lo is not None

    @torch.no_grad()
    def keep(self, xyz: torch.Tensor) -> torch.Tensor:
        """Bool mask [M]: True for points inside the bounds. Chunked so M x N distances never materialise."""
        xyz = xyz.detach()
        cams = self.cams.to(xyz.device)
        mask = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
        if self.lo is not None:
            lo, hi = self.lo.to(xyz.device), self.hi.to(xyz.device)
            mask &= ((xyz >= lo) & (xyz <= hi)).all(dim=1)
        if self.far_m > 0:
            far2 = self.far_m * self.far_m
            for s in range(0, len(xyz), CHUNK):
                pts = xyz[s : s + CHUNK].float()
                d2 = torch.cdist(pts, cams).pow_(2).min(dim=1).values if len(cams) else torch.zeros(len(pts), device=xyz.device)
                mask[s : s + CHUNK] &= d2 <= far2
        return mask

    def keep_np(self, xyz: np.ndarray) -> np.ndarray:
        return self.keep(torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32))).numpy()
