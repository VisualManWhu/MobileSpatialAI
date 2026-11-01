"""Capture format v1 loader. See docs/capture-format.md."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

SUPPORTED_VERSION = 1


class CaptureFormatError(Exception):
    def __init__(self, message_zh: str):
        super().__init__(message_zh)
        self.message_zh = message_zh


class Manifest(BaseModel):
    format_version: int
    capture_id: str
    room_id: str
    kind: str
    device: str
    created_at: str
    image_width: int
    image_height: int
    depth_width: int | None = None
    depth_height: int | None = None
    frame_count: int


class Frame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    t_ns: int
    image: str
    depth: str | None = None
    pose: np.ndarray  # 4x4 world_from_camera, OpenGL convention
    fx: float
    fy: float
    cx: float
    cy: float


class Capture(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest: Manifest
    frames: list[Frame]
    root: Path


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for name in zf.namelist():
        if not (dest_resolved / name).resolve().is_relative_to(dest_resolved):
            raise CaptureFormatError("拍摄包包含非法路径")
    zf.extractall(dest)


def load_capture(zip_path: Path, workdir: Path) -> Capture:
    zip_path, workdir = Path(zip_path), Path(workdir)
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as e:
        raise CaptureFormatError("拍摄包不是有效的 zip 文件") from e
    with zf:
        workdir.mkdir(parents=True, exist_ok=True)
        _safe_extract(zf, workdir)

    manifest_path = workdir / "manifest.json"
    frames_path = workdir / "frames.jsonl"
    if not manifest_path.is_file():
        raise CaptureFormatError("缺少 manifest.json")
    if not frames_path.is_file():
        raise CaptureFormatError("缺少 frames.jsonl")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CaptureFormatError("manifest.json 格式错误") from e
    version = raw.get("format_version") if isinstance(raw, dict) else None
    if version != SUPPORTED_VERSION:
        raise CaptureFormatError(f"不支持的拍摄包版本: {version}")
    try:
        manifest = Manifest(**raw)
    except ValidationError as e:
        raise CaptureFormatError(f"manifest.json 字段错误: {e.error_count()} 处") from e

    frames: list[Frame] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            pose = np.array(d["pose"], dtype=np.float64)
            if pose.size != 16:
                raise ValueError("pose")
            d["pose"] = pose.reshape(4, 4).T  # column-major -> row-major matrix
            frames.append(Frame(**d))
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError) as e:
            raise CaptureFormatError("frames.jsonl 格式错误") from e

    if len(frames) != manifest.frame_count:
        raise CaptureFormatError(
            f"帧数不一致: manifest 声明 {manifest.frame_count}, 实际 {len(frames)}"
        )

    for fr in frames:
        if not (workdir / fr.image).is_file():
            raise CaptureFormatError(f"缺少图片: {fr.image}")
        if fr.depth is not None and not (workdir / fr.depth).is_file():
            raise CaptureFormatError(f"缺少深度图: {fr.depth}")

    return Capture(manifest=manifest, frames=frames, root=workdir)
