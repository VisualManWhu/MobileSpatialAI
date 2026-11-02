"""Reconstruction profiles."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    max_image_side: int
    iterations: int
    max_gaussians: int
    sh_degree: int
    depth_lambda: float = 0.05  # 0 disables the depth loss
    pose_lr: float = 1e-4  # 0 disables per-camera pose refinement


PROFILES: dict[str, Profile] = {
    "laptop": Profile("laptop", 960, 15000, 1_500_000, 1),
    "server": Profile("server", 1600, 30000, 4_000_000, 3),
}
