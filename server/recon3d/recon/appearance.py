"""Per-frame affine colour correction: compensates auto-exposure / white-balance drift between frames.

The phone's AE/AWB make the same surface a different colour in different frames; without compensation
the model fits those differences with semi-transparent "fill light" blobs near walls and ceilings. Each
training frame i learns rgb' = A[i] @ rgb + b[i] (A initialised to the identity, b to zero), applied to
the *rendered* image before the photometric loss. Only the Gaussians are exported, so the correction is
dropped at export time: the splat keeps the colours of an "average" frame.

The per-frame deltas are mean-centred over all frames before use, so the average correction is exactly
the identity: the gauge freedom (every frame darkening while the Gaussians brighten) is removed and the
exported colours are those of the mean frame. An identity regulariser (reg()) in a first A/B left a global
drift that cost 1.3 dB on held-out views; it is kept only to damp outliers.
"""
from __future__ import annotations

import torch


class AffineColor(torch.nn.Module):
    """A[i] = I + dA[i] - mean(dA), b[i] = db[i] - mean(db) for frame i; both deltas zero-initialised."""

    def __init__(self, n_frames: int, device: str):
        super().__init__()
        self.dA = torch.nn.Parameter(torch.zeros(n_frames, 3, 3, device=device))
        self.db = torch.nn.Parameter(torch.zeros(n_frames, 3, device=device))

    def matrices(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(A [M,3,3], b [M,3]) for frames idx."""
        eye = torch.eye(3, dtype=self.dA.dtype, device=self.dA.device)
        dA = self.dA - self.dA.mean(dim=0, keepdim=True)
        db = self.db - self.db.mean(dim=0, keepdim=True)
        return eye + dA[idx], db[idx]

    def forward(self, rgb: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """rgb [M,H,W,3] rendered for frames idx [M] -> corrected rgb, same shape."""
        A, b = self.matrices(idx)
        return torch.einsum("mhwc,mdc->mhwd", rgb, A) + b[:, None, None, :]

    def reg(self) -> torch.Tensor:
        """mean over frames of ||A - I||_F^2 + ||b||^2."""
        return (self.dA**2).sum(dim=(1, 2)).mean() + (self.db**2).sum(dim=1).mean()

    @torch.no_grad()
    def stats(self) -> dict[str, float]:
        """Summary for logging: median / max gain deviation (||A - I||_F) and offset (||b||) of the centred corrections."""
        n = len(self.dA)
        A, b = self.matrices(torch.arange(n, device=self.dA.device))
        gain = (A - torch.eye(3, device=A.device)).flatten(1).norm(dim=1)
        off = b.norm(dim=1)
        return {
            "gain_med": float(gain.median()), "gain_max": float(gain.max()),
            "off_med": float(off.median()), "off_max": float(off.max()),
        }
