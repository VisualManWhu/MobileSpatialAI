"""Gaussian -> object assignment (spec 2026-11-08 §4.4): an object's 3D shape is its voxel set, so a
Gaussian belongs to the object whose (one-voxel dilated) voxel set contains its centre; structure
elements (floor, ceiling, walls, door / window openings) claim what is left by distance to their plane."""
from __future__ import annotations

import numpy as np

from recon3d.objects.segments import VOXEL_M, Segment, voxel_key
from recon3d.objects.structure import Room

PLANE_BAND_M = 0.05
OPENING_BAND_M = 0.15

_OFFSETS = np.array([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], dtype=np.int64)


def _pack(keys: np.ndarray) -> np.ndarray:
    """int64 voxel triplets -> one int64 per voxel (21 bits per axis, offset to stay positive)."""
    k = keys.astype(np.int64) + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def assign_by_voxels(positions: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """uint16 id per Gaussian (index into segments + 1, 0 = none); several hits -> the smallest volume."""
    n = len(positions)
    out = np.zeros(n, dtype=np.uint16)
    if n == 0 or not segments:
        return out
    keys = _pack(voxel_key(positions))
    best_vol = np.full(n, np.inf)
    for i, seg in enumerate(segments):
        if len(seg.voxels) == 0:
            continue
        dilated = (seg.voxels[:, None, :] + _OFFSETS[None, :, :]).reshape(-1, 3)
        packed = np.unique(_pack(dilated))
        hit = np.isin(keys, packed, assume_unique=False)
        vol = seg.volume()
        take = hit & (vol < best_vol)
        out[take] = i + 1
        best_vol[take] = vol
    return out


def assign_structure(
    positions: np.ndarray, room: Room, ids: np.ndarray, first_id: int, openings: list[tuple[int, Segment]]
) -> tuple[np.ndarray, list[int]]:
    """Fills Gaussians still at 0: door / window openings first (within OPENING_BAND_M of their wall
    plane and inside their bbox), then floor, ceiling and each wall (within PLANE_BAND_M, inside the
    extent). Returns the ids and the structure ids created in order [floor?, ceiling?, walls...];
    openings keep the ids they were given."""
    out = np.array(ids, dtype=np.uint16, copy=True)
    p = np.asarray(positions, dtype=np.float64)
    free = out == 0
    created: list[int] = []
    next_id = first_id

    for oid, seg in openings:
        inside = np.all((p >= seg.bbox_min) & (p <= seg.bbox_max), axis=1)
        near = np.zeros(len(p), dtype=bool)
        for wall in room.walls:
            near |= np.abs(room.wall_distance(p, wall)) < OPENING_BAND_M
        take = free & inside & near
        out[take] = oid
        free &= ~take

    if room.floor_y is not None:
        take = free & (np.abs(p[:, 1] - room.floor_y) < PLANE_BAND_M)
        out[take] = next_id
        free &= ~take
        created.append(next_id)
        next_id += 1
    if room.ceiling_y is not None:
        take = free & (np.abs(p[:, 1] - room.ceiling_y) < PLANE_BAND_M)
        out[take] = next_id
        free &= ~take
        created.append(next_id)
        next_id += 1
    for wall in room.walls:
        u = p @ wall.u
        u0, u1, y0, y1 = wall.extent
        take = free & (np.abs(room.wall_distance(p, wall)) < PLANE_BAND_M) & (u >= u0) & (u <= u1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        out[take] = next_id
        free &= ~take
        created.append(next_id)
        next_id += 1
    return out, created
