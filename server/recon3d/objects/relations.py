"""Geometric spatial relations between objects (on / in / next_to / left_of / right_of), pure numpy.
Computed here with rules, never guessed by the LLM (spec 2026-11-01 §5.4)."""
from __future__ import annotations

import numpy as np

from recon3d.objects.table import ObjectRecord

ON_GAP_M = 0.10  # a's bottom at most this far above b's top
ON_SINK_M = 0.02  # ... or this far below it (percentile boxes overlap a little)
IN_FRACTION = 0.9
NEXT_TO_GAP_M = 0.3
LR_MIN_SEP_M = 0.1


def _arr(o: ObjectRecord):
    return np.asarray(o.center, float), np.asarray(o.bbox_min, float), np.asarray(o.bbox_max, float)


def _horizontal_gap(a_min, a_max, b_min, b_max) -> float:
    gx = max(0.0, max(a_min[0], b_min[0]) - min(a_max[0], b_max[0]))
    gz = max(0.0, max(a_min[2], b_min[2]) - min(a_max[2], b_max[2]))
    return float(np.hypot(gx, gz))


def _inside_fraction(a_min, a_max, b_min, b_max) -> float:
    vol = float(np.prod(np.clip(a_max - a_min, 1e-9, None)))
    inter = float(np.prod(np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None)))
    return inter / vol


def compute_relations(
    objects: list[ObjectRecord], home_look: np.ndarray | None, max_dist_m: float = 1.5
) -> list[dict]:
    right = None
    if home_look is not None:
        look = np.array([home_look[0], 0.0, home_look[2]], float)
        if np.linalg.norm(look) > 1e-6:
            look /= np.linalg.norm(look)
            right = np.cross(look, np.array([0.0, 1.0, 0.0]))  # forward x up = right (+X when looking -Z)
    out: list[dict] = []
    geo = [_arr(o) for o in objects]
    for i, a in enumerate(objects):
        ca, amin, amax = geo[i]
        for j, b in enumerate(objects):
            if i == j:
                continue
            cb, bmin, bmax = geo[j]
            if np.linalg.norm(ca - cb) >= max_dist_m:
                continue
            # on: a's bottom near b's top and a's horizontal centre inside b's footprint
            if bmax[1] - ON_SINK_M <= amin[1] <= bmax[1] + ON_GAP_M and bmin[0] <= ca[0] <= bmax[0] and bmin[2] <= ca[2] <= bmax[2]:
                out.append({"a": a.id, "rel": "on", "b": b.id})
            if _inside_fraction(amin, amax, bmin, bmax) >= IN_FRACTION:
                out.append({"a": a.id, "rel": "in", "b": b.id})
            if i < j:
                vertical_overlap = min(amax[1], bmax[1]) > max(amin[1], bmin[1])
                if vertical_overlap and _horizontal_gap(amin, amax, bmin, bmax) < NEXT_TO_GAP_M:
                    out.append({"a": a.id, "rel": "next_to", "b": b.id})
            if right is not None:
                d = ca - cb
                sep = float(np.dot(np.array([d[0], 0.0, d[2]]), right))
                if abs(sep) >= LR_MIN_SEP_M:
                    out.append({"a": a.id, "rel": "right_of" if sep > 0 else "left_of", "b": b.id})
    return out
