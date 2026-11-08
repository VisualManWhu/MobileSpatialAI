"""Tracks (one SAM 2 identity over consecutive frames) -> 3D voxel segments, merging of re-appearing
objects, and support relations against the room structure (spec 2026-11-08 §4.2)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recon3d.objects.structure import Room

VOXEL_M = 0.02
MIN_TRACK_FRAMES = 5
MIN_VOXELS = 20
MIN_HITS = 3  # a voxel must be hit by this many frames of the track to survive
MERGE_OVERLAP = 0.5  # of the smaller segment's voxels
BEST_VIEWS = 5
BEST_VIEW_MIN_ANGLE_DEG = 20.0
SUPPORT_GAP_M = 0.10  # bottom-to-surface distance for floor / on
WALL_BAND_M = 0.15  # voxels within this of a wall plane count as "on the wall"
WALL_FRACTION = 0.6
SLICE_MIN_FRACTION = 0.02  # a 1-voxel slice with fewer than this of the peak slice's voxels is an outlier


@dataclass
class Track:
    """Produced by track.py; defined here so the CPU-only code and tests never import torch."""

    id: int
    frames: list[int]  # ascending frame indices
    masks: list[np.ndarray]  # np.packbits of the depth-resolution bool mask, parallel to frames
    boxes: list[tuple[float, float, float, float]]  # full-res pixel boxes, parallel to frames
    votes: dict[str, float] = field(default_factory=dict)  # proposer label -> summed score


@dataclass
class Segment:
    track_ids: list[int]
    frames: list[int]
    voxels: np.ndarray  # [k, 3] int64 voxel coordinates (floor(p / VOXEL_M))
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    size_lwh: tuple[float, float, float]  # horizontal min-area rectangle (l >= w), height
    votes: dict[str, float]
    best_views: list[tuple[int, tuple[float, float, float, float]]]  # (frame, full-res box)

    def volume(self) -> float:
        l, w, h = self.size_lwh
        return float(l * w * h)

    def centre(self) -> np.ndarray:
        return (self.bbox_min + self.bbox_max) / 2


def voxel_key(points: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(points, dtype=np.float64) / VOXEL_M).astype(np.int64)


def _unpack(mask_packed: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.unpackbits(mask_packed)[: h * w].reshape(h, w).astype(bool)


def _backproject(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    vs, us = np.nonzero(mask & (depth > 0))
    if len(vs) == 0:
        return np.zeros((0, 3))
    z = np.asarray(depth, dtype=np.float64)[vs, us]
    x = (us + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (vs + 0.5 - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (cam @ np.asarray(c2w, dtype=np.float64).T)[:, :3]


def _min_area_rect(xz: np.ndarray) -> tuple[float, float]:
    """(long, short) side of the minimum-area bounding rectangle of 2D points (1 degree sweep)."""
    if len(xz) == 0:
        return 0.0, 0.0
    best = None
    for deg in range(0, 90):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        u = xz[:, 0] * c + xz[:, 1] * s
        v = -xz[:, 0] * s + xz[:, 1] * c
        du, dv = np.ptp(u), np.ptp(v)
        if best is None or du * dv < best[0] * best[1]:
            best = (du, dv)
    long_, short = max(best), min(best)
    return float(long_), float(short)


def _robust_bounds(voxels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box of the voxel centres after dropping, per axis, the thin outlier slices: a slice
    (one voxel thick) with fewer than SLICE_MIN_FRACTION of the fullest slice's voxels. Unlike count
    percentiles this does not shave the sparsely seen bottom edge of an object whose top face dominates."""
    lo = np.zeros(3)
    hi = np.zeros(3)
    for ax in range(3):
        vals, counts = np.unique(voxels[:, ax], return_counts=True)
        keep = vals[counts >= max(1, SLICE_MIN_FRACTION * counts.max())]
        lo[ax] = (keep.min()) * VOXEL_M
        hi[ax] = (keep.max() + 1) * VOXEL_M
    return lo, hi


def _pick_best_views(track: Track, areas: np.ndarray, c2w_cv: np.ndarray) -> list[tuple[int, tuple]]:
    order = np.argsort(-areas)
    chosen: list[int] = []
    cos_lim = np.cos(np.radians(BEST_VIEW_MIN_ANGLE_DEG))
    for i in order:
        if areas[i] <= 0:
            break
        fwd = c2w_cv[track.frames[i]][:3, 2]
        if any(fwd @ c2w_cv[track.frames[j]][:3, 2] > cos_lim for j in chosen):
            continue
        chosen.append(int(i))
        if len(chosen) == BEST_VIEWS:
            break
    if not chosen and len(order):
        chosen = [int(order[0])]
    return [(track.frames[i], tuple(track.boxes[i])) for i in chosen]


def track_to_segment(track: Track, depths, Ks_small: np.ndarray, c2w_cv: np.ndarray, h: int, w: int) -> Segment | None:
    """Voxels hit by >= MIN_HITS frames of the track; None for short tracks or tracks without depth."""
    if len(track.frames) < MIN_TRACK_FRAMES:
        return None
    keys = []
    areas = np.zeros(len(track.frames))
    for i, (fi, packed) in enumerate(zip(track.frames, track.masks)):
        mask = _unpack(packed, h, w)
        areas[i] = mask.sum()
        pts = _backproject(mask, depths[fi], Ks_small[fi], c2w_cv[fi])
        if len(pts):
            keys.append(np.unique(voxel_key(pts), axis=0))  # one hit per frame per voxel
    if not keys:
        return None
    allk = np.concatenate(keys)
    uniq, counts = np.unique(allk, axis=0, return_counts=True)
    voxels = uniq[counts >= MIN_HITS]
    if len(voxels) < MIN_VOXELS:
        return None
    lo, hi = _robust_bounds(voxels)
    centres = (voxels + 0.5) * VOXEL_M
    inside = np.all((centres >= lo) & (centres <= hi), axis=1)
    core = centres[inside] if inside.any() else centres
    long_, short = _min_area_rect(core[:, [0, 2]])
    long_, short = long_ + VOXEL_M, short + VOXEL_M  # voxel centres -> outer faces
    size = (long_, short, float(hi[1] - lo[1]))
    return Segment(
        track_ids=[track.id],
        frames=list(track.frames),
        voxels=voxels,
        bbox_min=lo,
        bbox_max=hi,
        size_lwh=size,
        votes=dict(track.votes),
        best_views=_pick_best_views(track, areas, c2w_cv),
    )


def _voxel_set(seg: Segment) -> set[tuple[int, int, int]]:
    return set(map(tuple, seg.voxels.tolist()))


def _combine(a: Segment, b: Segment) -> Segment:
    voxels = np.unique(np.concatenate([a.voxels, b.voxels]), axis=0)
    lo, hi = _robust_bounds(voxels)
    centres = (voxels + 0.5) * VOXEL_M
    long_, short = _min_area_rect(centres[:, [0, 2]])
    long_, short = long_ + VOXEL_M, short + VOXEL_M
    votes = dict(a.votes)
    for k, v in b.votes.items():
        votes[k] = votes.get(k, 0.0) + v
    views = (a.best_views + b.best_views)[:BEST_VIEWS]
    return Segment(
        track_ids=a.track_ids + b.track_ids,
        frames=sorted(set(a.frames) | set(b.frames)),
        voxels=voxels,
        bbox_min=lo,
        bbox_max=hi,
        size_lwh=(long_, short, float(hi[1] - lo[1])),
        votes=votes,
        best_views=views,
    )


def merge_segments(segments: list[Segment]) -> list[Segment]:
    """Greedy: two segments with no frame in common whose voxel overlap is >= MERGE_OVERLAP of the
    smaller one are the same object seen again. Segments that share a frame are never merged."""
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: -len(s.voxels)):
        sv = _voxel_set(seg)
        sf = set(seg.frames)
        for i, other in enumerate(out):
            if sf & set(other.frames):
                continue
            ov = _voxel_set(other)
            overlap = len(sv & ov) / max(1, min(len(sv), len(ov)))
            if overlap >= MERGE_OVERLAP:
                out[i] = _combine(other, seg)
                break
        else:
            out.append(seg)
    return out


def support_of(seg: Segment, room: Room, others: list[Segment]) -> str:
    """floor | ceiling | on:<index into others> | wall:<id> | none, in that priority."""
    bottom, top = float(seg.bbox_min[1]), float(seg.bbox_max[1])
    if room.floor_y is not None and abs(bottom - room.floor_y) < SUPPORT_GAP_M:
        return "floor"
    if room.ceiling_y is not None and abs(top - room.ceiling_y) < SUPPORT_GAP_M:
        return "ceiling"
    cx, cz = seg.centre()[[0, 2]]
    best = None
    for i, o in enumerate(others):
        if o is seg:
            continue
        gap = bottom - float(o.bbox_max[1])
        if -SUPPORT_GAP_M / 2 <= gap < SUPPORT_GAP_M and o.bbox_min[0] <= cx <= o.bbox_max[0] and o.bbox_min[2] <= cz <= o.bbox_max[2]:
            if best is None or gap < best[0]:
                best = (gap, i)
    if best is not None:
        return f"on:{best[1]}"
    if room.walls:
        centres = (seg.voxels + 0.5) * VOXEL_M
        for wall in room.walls:
            frac = (np.abs(room.wall_distance(centres, wall)) < WALL_BAND_M).mean()
            if frac >= WALL_FRACTION:
                return f"wall:{wall.id}"
    return "none"
