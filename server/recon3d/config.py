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
    # Depth loss is opt-in (0 = off): on three real captures, including one with raw-depth confidence,
    # it changed the floater score by < 0.002 and PSNR by < 0.1 dB while costing +40% training time
    # (docs/acceptance/phase1.md section 5). The confidence maps act through init_points instead.
    depth_lambda: float = 0.0
    # Pose refinement is opt-in (0 = off): on the office capture it lost 2.2 dB PSNR, removed fewer floaters
    # and cost +50% time (docs/acceptance/phase1.md section 5).
    pose_lr: float = 0.0


PROFILES: dict[str, Profile] = {
    "laptop": Profile("laptop", 960, 15000, 1_500_000, 1),
    "server": Profile("server", 1600, 30000, 4_000_000, 3),
}
