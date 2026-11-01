"""Build synthetic capture-format-v1 zips for tests."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def _default_pose(i: int) -> list[float]:
    # column-major 4x4, translation along x by 0.1*i
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1 * i, 0, 0, 1]


def make_capture(
    tmp_path: Path,
    n_frames: int = 4,
    with_depth: bool = True,
    *,
    format_version: int = 1,
    drop_image: int | None = None,
    frame_count: int | None = None,
    image_width: int = 64,
    image_height: int = 48,
    depth_width: int | None = None,
    depth_height: int | None = None,
    fx: float = 50.0,
    fy: float = 50.0,
    cx: float | None = None,
    cy: float | None = None,
    poses: list[list[float]] | None = None,
    images: list[np.ndarray] | None = None,
    depths: list[np.ndarray] | None = None,
    name: str = "capture.zip",
) -> Path:
    """Write a capture zip into tmp_path and return its path.

    drop_image: 1-based frame index whose image file is omitted.
    frame_count: overrides the manifest value only.
    poses: optional list of 16-float column-major poses (one per frame).
    images: optional list of uint8 HxWx3 arrays (one per frame) instead of flat colors;
        image_width/height are taken from the first one.
    depths: optional list of uint16 HxW arrays in millimeters (one per frame); depth
        width/height are taken from the first one and with_depth is implied.
    """
    if images is not None:
        image_height, image_width = images[0].shape[:2]
    if depths is not None:
        with_depth = True
        depth_height, depth_width = depths[0].shape[:2]
    dw = depth_width or max(1, image_width // 8)
    dh = depth_height or max(1, image_height // 8)
    cx = image_width / 2 if cx is None else cx
    cy = image_height / 2 if cy is None else cy
    manifest = {
        "format_version": format_version,
        "capture_id": "cap-test",
        "room_id": "room-test",
        "kind": "model",
        "device": "test-device",
        "created_at": "2026-11-01T00:00:00Z",
        "image_width": image_width,
        "image_height": image_height,
        "depth_width": dw if with_depth else None,
        "depth_height": dh if with_depth else None,
        "frame_count": n_frames if frame_count is None else frame_count,
    }
    out = Path(tmp_path) / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        lines = []
        for i in range(1, n_frames + 1):
            img_name = f"images/{i:06d}.jpg"
            depth_name = f"depth/{i:06d}.png" if with_depth else None
            pose = poses[i - 1] if poses is not None else _default_pose(i)
            lines.append(json.dumps({
                "index": i, "t_ns": i * 33_000_000, "image": img_name,
                "depth": depth_name, "pose": pose,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }))
            if drop_image != i:
                buf = io.BytesIO()
                if images is not None:
                    Image.fromarray(images[i - 1]).save(buf, "JPEG", quality=95)
                else:
                    Image.new("RGB", (image_width, image_height),
                              (40 * i % 256, 100, 160)).save(buf, "JPEG")
                zf.writestr(img_name, buf.getvalue())
            if with_depth:
                arr = (depths[i - 1] if depths is not None
                       else np.full((dh, dw), 1500, dtype=np.uint16))
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, "PNG")
                zf.writestr(depth_name, buf.getvalue())
        zf.writestr("frames.jsonl", "\n".join(lines) + "\n")
    return out
