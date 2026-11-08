"""The object table: objects.json + objects.bin (+ thumbs/) zipped into objects.zip."""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERSION = 2  # v2 adds level / size / support / confidence per object and a structure block


@dataclass(frozen=True)
class ObjectRecord:
    id: int
    label: str
    label_zh: str
    score: float
    views: int
    center: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    gaussian_count: int
    level: str = "item"  # structure | furniture | item
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)  # l, w, h metres (gravity-aligned)
    support: str = "none"  # floor | ceiling | on:<id> | wall:<id> | none
    confidence: float = 0.0


def encode_ids(ids: np.ndarray) -> bytes:
    """Per-Gaussian object id (0 = none) as little-endian uint16, in scene.splat order."""
    return np.ascontiguousarray(ids, dtype="<u2").tobytes()


def decode_ids(data: bytes) -> np.ndarray:
    if len(data) % 2:
        raise ValueError(f"objects.bin has an odd length ({len(data)} bytes)")
    return np.frombuffer(data, dtype="<u2").astype(np.uint16)


def _f3(v) -> list[float]:
    return [float(x) for x in v]


def write_objects(
    out_dir: Path,
    job_id: str,
    frames_used: int,
    ids: np.ndarray,
    objects: list[ObjectRecord],
    relations: list[dict],
    structure: dict | None = None,
) -> Path:
    """Writes objects.json and objects.bin into out_dir (thumbs/ must already be there) and zips the
    directory into out_dir.parent / objects.zip. Returns the zip path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "version": VERSION,
        "job_id": job_id,
        "frames_used": int(frames_used),
        "gaussians": int(len(ids)),
        "objects": [
            {
                "id": o.id,
                "label": o.label,
                "label_zh": o.label_zh,
                "score": round(float(o.score), 4),
                "views": int(o.views),
                "center": _f3(o.center),
                "bbox": {"min": _f3(o.bbox_min), "max": _f3(o.bbox_max)},
                "gaussian_count": int(o.gaussian_count),
                "thumb": f"thumbs/{o.id}.jpg",
                "level": o.level,
                "size": _f3(o.size),
                "support": o.support,
                "confidence": round(float(o.confidence), 4),
            }
            for o in objects
        ],
        "relations": relations,
        "structure": structure if structure is not None else {"floor_y": None, "ceiling_y": None, "walls": []},
    }
    (out_dir / "objects.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "objects.bin").write_bytes(encode_ids(ids))
    zip_path = out_dir.parent / "objects.zip"
    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                mode = zipfile.ZIP_DEFLATED if p.suffix == ".json" else zipfile.ZIP_STORED
                zf.write(p, p.relative_to(out_dir).as_posix(), compress_type=mode)
    tmp.replace(zip_path)
    return zip_path


def read_objects_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
