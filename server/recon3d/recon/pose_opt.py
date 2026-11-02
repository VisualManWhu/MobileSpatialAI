"""Per-camera pose refinement: a small se(3) correction applied in the camera frame."""
from __future__ import annotations

import torch

SMALL_ANGLE = 1e-6


def _hat(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrices [M,3,3] of vectors [M,3]."""
    z = torch.zeros_like(v[:, 0])
    x, y, w = v[:, 0], v[:, 1], v[:, 2]
    return torch.stack([z, -w, y, w, z, -x, -y, x, z], dim=1).view(-1, 3, 3)


def se3_exp(omega: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """[M,4,4] transforms with rotation exp(omega) (Rodrigues) and translation t.

    The translation is applied as-is (not the V-matrix of the full SE(3) exponential):
    for corrections this small the two parameterisations are equivalent to optimise.
    """
    theta = omega.norm(dim=1)
    small = theta < SMALL_ANGLE
    th = torch.where(small, torch.ones_like(theta), theta)  # avoid 0/0 (grad-safe)
    a = torch.where(small, torch.ones_like(theta), torch.sin(th) / th)
    b = torch.where(small, torch.full_like(theta, 0.5), (1.0 - torch.cos(th)) / th**2)
    Wx = _hat(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(Wx)
    R = eye + a[:, None, None] * Wx + b[:, None, None] * (Wx @ Wx)
    T = torch.eye(4, dtype=omega.dtype, device=omega.device).repeat(len(omega), 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


class PoseRefine(torch.nn.Module):
    """delta[i] = [omega(3), t(3)] for camera i, zero-initialised."""

    def __init__(self, n_cams: int, device: str):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(n_cams, 6, device=device))

    def forward(self, viewmats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Left-multiplies each viewmat by exp(omega, t) (camera-frame correction)."""
        d = self.delta[idx]
        return se3_exp(d[:, :3], d[:, 3:]) @ viewmats

    def reg(self) -> torch.Tensor:
        """mean of ||delta||^2"""
        return (self.delta**2).sum(dim=1).mean()
