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
    # Scene bounds (recon/bounds.py): a Gaussian survives only within far_m of the nearest camera position
    # and inside the camera bounding box grown by bbox_margin_m. Far depth is noise and Gaussians that
    # drift out of the room are floaters. 0 disables a criterion. Applied to the initial cloud, at every
    # densification step and once more at the end of training.
    far_m: float = 5.0
    bbox_margin_m: float = 5.0
    # Floater pruning (recon/contrib.py, recon/outliers.py). Contribution pruning renders every training
    # view at half resolution and removes Gaussians visible (> 1 blended pixel at half resolution) in fewer
    # than contrib_min_views views, or whose total contribution is below contrib_min_px; it runs at iterations // 2,
    # 3 * iterations // 4 and once after training (both thresholds 0 disables it). The kNN statistical
    # outlier filter (k = 8) then removes isolated Gaussian centres beyond mean + sor_sigma * std (0 disables).
    contrib_min_views: int = 3
    contrib_min_px: float = 0.5
    sor_sigma: float = 3.0
    # Regularisers added to the loss every step (trainer.reg_loss): opacity_reg * mean(opacity) pushes
    # unneeded Gaussians towards transparency so they get pruned; scale_reg * mean(scale) discourages
    # huge blobs. 0 disables each.
    opacity_reg: float = 0.01
    scale_reg: float = 0.01


PROFILES: dict[str, Profile] = {
    "laptop": Profile("laptop", 960, 15000, 1_500_000, 1),
    "server": Profile("server", 1600, 30000, 4_000_000, 3),
}
