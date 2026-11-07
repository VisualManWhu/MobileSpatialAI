"""Per-frame NMS and the greedy 3D merge of per-frame detections into objects. Pure numpy/scipy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Detection:
    frame: int
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels
    points: np.ndarray  # [M, 3] world-frame points of the mask (may be empty for NMS-only use)


@dataclass
class MergedObject:
    label: str
    score: float
    views: int
    center: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    members: list[int]  # indices into the input detections


def _box_iou_2d(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def nms_frame(dets: list[Detection], iou: float = 0.7) -> list[Detection]:
    """Cross-class NMS inside one frame (prompt batches return duplicate boxes): keep the higher score."""
    order = sorted(dets, key=lambda d: -d.score)
    kept: list[Detection] = []
    for d in order:
        if all(_box_iou_2d(d.box, k.box) < iou for k in kept):
            kept.append(d)
    return kept


def point_overlap(a: np.ndarray, b: np.ndarray, radius_m: float) -> float:
    """Fraction of points of `a` within radius_m of some point of `b`."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d, _ = cKDTree(b).query(a, k=1, distance_upper_bound=radius_m)
    return float(np.mean(np.isfinite(d)))


def bbox_iou(a_min, a_max, b_min, b_max) -> float:
    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    union = float(np.prod(a_max - a_min)) + float(np.prod(b_max - b_min)) - inter
    return inter / union if union > 0 else 0.0


def _percentile_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(points, 5, axis=0), np.percentile(points, 95, axis=0)


def merge_detections(
    dets: list[Detection],
    same_overlap: float = 0.3,
    same_iou: float = 0.3,
    cross_overlap: float = 0.6,
    cross_iou: float = 0.5,
    radius_m: float = 0.05,
    min_views: int = 2,
    max_center_dist_m: float = 2.0,
) -> list[MergedObject]:
    """Greedy union-find: two detections from different frames merge when their point clouds overlap
    (same label: >= same_overlap OR bbox IoU >= same_iou; different labels: >= cross_overlap AND
    IoU >= cross_iou). Detections of the same frame never merge. Groups seen in fewer than min_views
    frames are dropped; the label is the one with the largest score sum inside the group."""
    n = len(dets)
    parent = list(range(n))
    frames_of: list[set[int]] = [{d.frame} for d in dets]  # per root: frames already in the group

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri == rj or frames_of[ri] & frames_of[rj]:
            return  # a group never holds two detections of one frame (two cups side by side stay two)
        parent[rj] = ri
        frames_of[ri] |= frames_of[rj]

    boxes = [_percentile_box(d.points) if len(d.points) else (np.zeros(3), np.zeros(3)) for d in dets]
    centers = [np.median(d.points, axis=0) if len(d.points) else np.zeros(3) for d in dets]
    trees = [cKDTree(d.points) if len(d.points) else None for d in dets]

    def overlap(i: int, j: int) -> float:
        if trees[i] is None or trees[j] is None:
            return 0.0
        di, _ = trees[j].query(dets[i].points, k=1, distance_upper_bound=radius_m)
        dj, _ = trees[i].query(dets[j].points, k=1, distance_upper_bound=radius_m)
        return max(float(np.mean(np.isfinite(di))), float(np.mean(np.isfinite(dj))))

    for i in range(n):
        for j in range(i + 1, n):
            if dets[i].frame == dets[j].frame or find(i) == find(j):
                continue
            if np.linalg.norm(centers[i] - centers[j]) > max_center_dist_m:
                continue
            iou = bbox_iou(boxes[i][0], boxes[i][1], boxes[j][0], boxes[j][1])
            if dets[i].label == dets[j].label:
                ok = iou >= same_iou or overlap(i, j) >= same_overlap
            else:
                ok = iou >= cross_iou and overlap(i, j) >= cross_overlap
            if ok:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[MergedObject] = []
    for members in groups.values():
        frames = {dets[i].frame for i in members}
        if len(frames) < min_views:
            continue
        votes: dict[str, float] = {}
        for i in members:
            votes[dets[i].label] = votes.get(dets[i].label, 0.0) + dets[i].score
        label = max(votes, key=votes.get)
        pts = np.concatenate([dets[i].points for i in members if len(dets[i].points)]) if any(
            len(dets[i].points) for i in members
        ) else np.zeros((1, 3))
        lo, hi = _percentile_box(pts)
        out.append(
            MergedObject(
                label=label,
                score=max(dets[i].score for i in members),
                views=len(frames),
                center=np.median(pts, axis=0),
                bbox_min=lo,
                bbox_max=hi,
                members=sorted(members),
            )
        )
    out.sort(key=lambda o: (-o.views, -o.score))
    return out
