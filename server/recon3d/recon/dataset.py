"""Training dataset built from a loaded capture."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from recon3d.capture.format import Capture
from recon3d.capture.poses import gl_to_cv_c2w
from recon3d.config import Profile

log = logging.getLogger(__name__)

ASPECT_TOLERANCE = 0.01


@dataclass
class TrainingSet:
    """Training tensors, OpenCV camera convention.

    images: [N,H,W,3] float32 in 0..1 (downscaled so max side <= profile.max_image_side).
    K: [N,3,3] intrinsics matching the downscaled images.
    c2w: [N,4,4] camera-to-world, OpenCV convention.
    depths: per-frame [dh,dw] float32 tensor in meters (0 = invalid) at the native
        depth resolution, or None (also None when the depth was skipped because its
        aspect ratio does not match the image and no depth intrinsics were given).
    depth_K: optional per-frame [3,3] intrinsics in depth-image pixels (or None per frame).
        When absent/None, back-projection uses the image K scaled by dw/W and dh/H.
    """

    images: torch.Tensor
    K: torch.Tensor
    c2w: torch.Tensor
    depths: list[torch.Tensor | None]
    depth_K: list[torch.Tensor | None] | None = field(default=None)


def build_training_set(cap: Capture, profile: Profile, device: str = "cpu") -> TrainingSet:
    m = cap.manifest
    scale = min(1.0, profile.max_image_side / max(m.image_width, m.image_height))
    w = max(1, round(m.image_width * scale))
    h = max(1, round(m.image_height * scale))
    sx, sy = w / m.image_width, h / m.image_height

    images, Ks, poses, depths, depth_Ks = [], [], [], [], []
    for fr in cap.frames:
        with Image.open(cap.root / fr.image) as im:
            im = im.convert("RGB")
            if im.size != (w, h):
                im = im.resize((w, h), Image.LANCZOS)
            images.append(np.asarray(im, dtype=np.float32) / 255.0)
        Ks.append([[fr.fx * sx, 0, fr.cx * sx], [0, fr.fy * sy, fr.cy * sy], [0, 0, 1]])
        poses.append(gl_to_cv_c2w(fr.pose))
        if fr.depth is None:
            depths.append(None)
            depth_Ks.append(None)
            continue
        with Image.open(cap.root / fr.depth) as d:
            arr = np.asarray(d, dtype=np.float32) / 1000.0  # 16-bit mm -> m
        dh, dw = arr.shape[:2]
        dk = (fr.depth_fx, fr.depth_fy, fr.depth_cx, fr.depth_cy)
        if all(v is not None for v in dk):
            dfx, dfy, dcx, dcy = dk
            depth_Ks.append(torch.tensor([[dfx, 0, dcx], [0, dfy, dcy], [0, 0, 1]],
                                         dtype=torch.float32, device=device))
        else:
            ratio = (dw / dh) / (m.image_width / m.image_height)
            if abs(ratio - 1.0) > ASPECT_TOLERANCE:
                log.warning(
                    "frame %d: depth aspect %dx%d does not match image %dx%d and no depth "
                    "intrinsics given; skipping depth for this frame",
                    fr.index, dw, dh, m.image_width, m.image_height,
                )
                depths.append(None)
                depth_Ks.append(None)
                continue
            depth_Ks.append(None)
        depths.append(torch.from_numpy(arr).to(device))

    return TrainingSet(
        images=torch.from_numpy(np.stack(images)).to(device),
        K=torch.tensor(Ks, dtype=torch.float32, device=device),
        c2w=torch.tensor(np.stack(poses), dtype=torch.float32, device=device),
        depths=depths,
        depth_K=depth_Ks,
    )
