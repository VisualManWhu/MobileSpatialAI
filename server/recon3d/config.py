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
    # Depth-cloud support (recon/bounds.py): a Gaussian farther than support_far_m from every camera must lie
    # within support_dist_m of a point of the confidence-filtered depth init cloud, or it goes. Far from the
    # cameras that cloud is the only evidence of geometry; on the far_m = 5 -> 3 experiment the floaters
    # just moved into the 2-3 m band (docs/acceptance/phase1.md section 5), so a hard radius cannot fix
    # them. Applied with the other bounds criteria. 0 in either value disables it. The lookup is a voxel
    # hash with cell = support_dist_m, so up to 2 * sqrt(3) * support_dist_m counts as supported.
    support_far_m: float = 2.0
    support_dist_m: float = 0.3
    # Floater pruning (recon/contrib.py, recon/outliers.py), opt-in (0 = off). Contribution pruning renders
    # every training view at half resolution and removes Gaussians visible (> contrib_view_px blended pixels
    # at half resolution) in fewer than contrib_min_views views, or whose total contribution is below
    # contrib_min_px; it runs at iterations // 2, 3 * iterations // 4 and once after training (both
    # thresholds 0 disables it). The kNN statistical outlier filter (k = 8) then removes isolated Gaussian
    # centres beyond mean + sor_sigma * std (0 disables). Evidence: on two real captures contribution
    # pruning cost 1.3 to 4 dB PSNR at every threshold tried (docs/acceptance/phase1.md section 5);
    # the old opt-in values were contrib_min_views=3, contrib_min_px=0.5.
    contrib_min_views: int = 0
    contrib_min_px: float = 0.0
    # A Gaussian counts as visible in a view when its blended weight there exceeds this many
    # half-resolution pixels (the per-view threshold behind contrib_min_views).
    contrib_view_px: float = 1.0
    # SOR is off: its effect was never isolated in the A/B (docs/acceptance/phase1.md section 5);
    # the old opt-in value was 3.0.
    sor_sigma: float = 0.0
    # Regularisers added to the loss every step (trainer.reg_loss): opacity_reg * mean(opacity) pushes
    # unneeded Gaussians towards transparency so they get pruned; scale_reg * mean(scale) discourages
    # huge blobs. 0 disables each. Measured together with MCMC (see strategy below).
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    # Densification strategy: "default" (gsplat DefaultStrategy: clone/split/prune plus our cap and opacity
    # reset) or "mcmc" (gsplat MCMCStrategy: relocates dead Gaussians and grows up to max_gaussians).
    # MCMC: contribution-pruning passes during relocation (step < refine_stop_iter) lower opacity so relocation
    # can reuse those Gaussians; passes at or after refine_stop_iter remove, as with "default".
    # Evidence: MCMC + the regularisers above, on two real captures, outdoor 9f6d PSNR 18.71 -> 19.10 dB and
    # floater score 0.375 -> 0.116, indoor 5cba 23.59 -> 24.30 dB and opaque floaters 57.6k -> 16.2k
    # (docs/acceptance/phase1.md section 5).
    strategy: str = "mcmc"


PROFILES: dict[str, Profile] = {
    "laptop": Profile("laptop", 960, 15000, 1_500_000, 1),
    "server": Profile("server", 1600, 30000, 4_000_000, 3),
}
