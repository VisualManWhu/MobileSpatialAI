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


PROFILES: dict[str, Profile] = {
    "laptop": Profile("laptop", 960, 15000, 1_500_000, 1),
    "server": Profile("server", 1600, 30000, 4_000_000, 3),
}
