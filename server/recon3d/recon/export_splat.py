"""Export trained Gaussians to the compact .splat format and the full 3DGS .ply."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from recon3d.recon.trainer import SH_C0, Splats

SPLAT_DTYPE = np.dtype(
    [("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)]
)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def export_splat(splats: Splats, path: Path, min_opacity: float = 0.005) -> int:
    """Write 32-byte-per-Gaussian .splat; returns the number of Gaussians written."""
    means = _np(splats.means)
    scales = _np(splats.scales)
    quats = _np(splats.quats)
    alpha = 1.0 / (1.0 + np.exp(-_np(splats.opacities).reshape(-1).astype(np.float64)))
    sh0 = _np(splats.sh0).reshape(-1, 3)

    keep = alpha >= min_opacity
    means, scales, quats, alpha, sh0 = (a[keep] for a in (means, scales, quats, alpha, sh0))

    importance = np.exp(scales.astype(np.float64).sum(axis=1)) * alpha
    order = np.argsort(-importance, kind="stable")
    means, scales, quats, alpha, sh0 = (a[order] for a in (means, scales, quats, alpha, sh0))

    n = len(means)
    out = np.empty(n, dtype=SPLAT_DTYPE)
    out["pos"] = means
    out["scale"] = np.exp(scales)
    rgb = np.clip(0.5 + SH_C0 * sh0, 0.0, 1.0) * 255.0
    out["rgba"][:, :3] = np.round(rgb).astype(np.uint8)
    out["rgba"][:, 3] = np.round(alpha * 255.0).astype(np.uint8)
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    q = quats / np.maximum(norm, 1e-12)
    out["rot"] = np.clip(np.trunc(q * 128.0 + 128.0), 0, 255).astype(np.uint8)

    Path(path).write_bytes(out.tobytes())
    return n


def export_ply(splats: Splats, path: Path) -> int:
    """Write the full model as a standard 3DGS binary PLY (raw parameters, no filtering)."""
    means = _np(splats.means)
    n = len(means)
    sh0 = _np(splats.sh0).reshape(n, 3)
    shN = _np(splats.shN).reshape(n, -1, 3)
    k = shN.shape[1]
    # reference ordering: channel-major -> f_rest index = c * K + j
    rest = shN.transpose(0, 2, 1).reshape(n, 3 * k)
    cols = [
        means,
        np.zeros((n, 3), np.float32),
        sh0,
        rest,
        _np(splats.opacities).reshape(n, 1),
        _np(splats.scales),
        _np(splats.quats),
    ]
    data = np.concatenate(cols, axis=1).astype("<f4", copy=False)

    names = (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(3 * k)]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    assert data.shape[1] == len(names)
    header = "ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())
    return n


def load_ply(path: Path, device: str = "cpu") -> Splats:
    """Inverse of export_ply: read a 3DGS binary PLY back into raw Splats parameters."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode("ascii").splitlines()
    if lines[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError(f"{path}: not a binary little-endian PLY")
    n = next(int(l.split()[2]) for l in lines if l.startswith("element vertex "))
    names = []
    for l in lines:
        if l.startswith("property "):
            parts = l.split()
            if parts[1] != "float":
                raise ValueError(f"{path}: unsupported property type {parts[1]!r}")
            names.append(parts[2])
    col = {nm: i for i, nm in enumerate(names)}
    data = np.frombuffer(raw, "<f4", count=n * len(names), offset=end).reshape(n, len(names))

    def cols(*nms: str) -> np.ndarray:
        return data[:, [col[nm] for nm in nms]]

    n_rest = sum(nm.startswith("f_rest_") for nm in names)
    k = n_rest // 3
    rest = cols(*(f"f_rest_{i}" for i in range(n_rest))).reshape(n, 3, k).transpose(0, 2, 1)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(a, dtype=np.float32)).to(device)  # copy: buffer is read-only

    return Splats(
        means=t(cols("x", "y", "z")),
        scales=t(cols("scale_0", "scale_1", "scale_2")),
        quats=t(cols("rot_0", "rot_1", "rot_2", "rot_3")),
        opacities=t(data[:, col["opacity"]]),
        sh0=t(cols("f_dc_0", "f_dc_1", "f_dc_2").reshape(n, 1, 3)),
        shN=t(rest),
    )
