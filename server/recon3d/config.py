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
    # Pose refinement is opt-in (0 = off): on the office capture it lost 2.2 dB PSNR, removed fewer floaters
    # and cost +50% time (docs/acceptance/phase1.md section 5).
    pose_lr: float = 0.0


PROFILES: dict[str, Profile] = {
    "laptop": Profile("laptop", 960, 15000, 1_500_000, 1),
    "server": Profile("server", 1600, 30000, 4_000_000, 3),
}
