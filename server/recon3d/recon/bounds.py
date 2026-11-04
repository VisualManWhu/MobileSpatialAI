"""Scene bounds derived from the camera path and the depth init cloud: everything far from every camera is noise.

Depth maps beyond a few metres are unreliable and Gaussians that densify or drift out of the room
render as floaters, so a Gaussian is kept only if it is
  * within ``far_m`` of the nearest camera position,
  * inside the camera-position bounding box grown by ``bbox_margin_m`` on each side, and
  * supported: either within ``support_far_m`` of the nearest camera, or within ``support_dist_m``
    of some point of the (confidence-filtered) depth init cloud. Far from the cameras the depth cloud
    is the only evidence of geometry, so a Gaussian with no depth point nearby is a floater.
Each criterion is disabled with a value of 0 (support: either of its two values, or no cloud).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

CHUNK = 65_536
# Voxel keys are packed into int64 with this many bits per axis; the cloud spans at most 2^21 cells.
_KEY_BITS = 21
_KEY_OFF = 1 << (_KEY_BITS - 1)


def _voxel_keys(xyz: torch.Tensor, cell: float) -> torch.Tensor:
    ijk = torch.floor(xyz / cell).long() + _KEY_OFF
    ijk = ijk.clamp(0, (1 << _KEY_BITS) - 1)
    return (ijk[:, 0] << (2 * _KEY_BITS)) | (ijk[:, 1] << _KEY_BITS) | ijk[:, 2]


@dataclass(frozen=True)
class SceneBounds:
    cams: torch.Tensor  # [N, 3] camera positions (world)
    far_m: float
    lo: torch.Tensor | None  # [3] grown bbox min, None = off
    hi: torch.Tensor | None  # [3]
    support_far_m: float = 0.0
    support_dist_m: float = 0.0
    support_keys: torch.Tensor | None = None  # sorted unique voxel keys of the depth cloud, None = off

    @classmethod
    def from_c2w(
        cls,
        c2w: torch.Tensor,
        far_m: float,
        bbox_margin_m: float,
        support_xyz: np.ndarray | torch.Tensor | None = None,
        support_far_m: float = 0.0,
        support_dist_m: float = 0.0,
    ) -> "SceneBounds":
        cams = c2w[:, :3, 3].detach().float()
        lo = hi = None
        if bbox_margin_m > 0:
            lo = cams.min(dim=0).values - bbox_margin_m
            hi = cams.max(dim=0).values + bbox_margin_m
        keys = None
        if support_xyz is not None and support_far_m > 0 and support_dist_m > 0 and len(support_xyz) > 0:
            pts = torch.as_tensor(np.asarray(support_xyz) if isinstance(support_xyz, np.ndarray) else support_xyz)
            keys = torch.unique(_voxel_keys(pts.detach().float().cpu(), support_dist_m))
        else:
            support_far_m = support_dist_m = 0.0
        return cls(
            cams=cams, far_m=float(far_m), lo=lo, hi=hi,
            support_far_m=float(support_far_m), support_dist_m=float(support_dist_m), support_keys=keys,
        )

    @property
    def enabled(self) -> bool:
        return self.far_m > 0 or self.lo is not None or self.support_keys is not None

    @torch.no_grad()
    def _supported(self, pts: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """True where some depth point lies in the point's voxel or one of the 26 neighbours.

        With cell == support_dist_m every point within support_dist_m of a depth point is found;
        points up to 2 * sqrt(3) * support_dist_m away may also pass (the test is conservative).
        """
        base = torch.floor(pts / self.support_dist_m).long() + _KEY_OFF
        hit = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    ijk = (base + torch.tensor([dx, dy, dz], device=pts.device)).clamp(0, (1 << _KEY_BITS) - 1)
                    k = (ijk[:, 0] << (2 * _KEY_BITS)) | (ijk[:, 1] << _KEY_BITS) | ijk[:, 2]
                    pos = torch.searchsorted(keys, k).clamp_max(len(keys) - 1)
                    hit |= keys[pos] == k
        return hit

    @torch.no_grad()
    def keep(self, xyz: torch.Tensor) -> torch.Tensor:
        """Bool mask [M]: True for points inside the bounds. Chunked so M x N distances never materialise."""
        xyz = xyz.detach()
        cams = self.cams.to(xyz.device)
        mask = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
        if self.lo is not None:
            lo, hi = self.lo.to(xyz.device), self.hi.to(xyz.device)
            mask &= ((xyz >= lo) & (xyz <= hi)).all(dim=1)
        keys = self.support_keys.to(xyz.device) if self.support_keys is not None else None
        if self.far_m > 0 or keys is not None:
            far2 = self.far_m * self.far_m
            sup2 = self.support_far_m * self.support_far_m
            for s in range(0, len(xyz), CHUNK):
                pts = xyz[s : s + CHUNK].float()
                d2 = torch.cdist(pts, cams).pow_(2).min(dim=1).values if len(cams) else torch.zeros(len(pts), device=xyz.device)
                if self.far_m > 0:
                    mask[s : s + CHUNK] &= d2 <= far2
                if keys is not None:
                    mask[s : s + CHUNK] &= (d2 <= sup2) | self._supported(pts, keys)
        return mask

    def keep_np(self, xyz: np.ndarray) -> np.ndarray:
        return self.keep(torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32))).numpy()
