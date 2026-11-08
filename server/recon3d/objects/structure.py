"""Room structure (floor, ceiling, walls) by RANSAC plane fitting on the rendered-depth point cloud.

World frame is ARCore's: +Y up, poses gravity-aligned, so horizontal / vertical planes are read off
the normal's angle to +Y directly (spec 2026-11-08 §4.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_INLIER_M = 0.03
UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class Wall:
    id: int
    normal: np.ndarray  # unit, horizontal, pointing into the room
    d: float  # plane: normal · p + d = 0
    u: np.ndarray  # horizontal in-plane unit axis
    extent: tuple[float, float, float, float]  # u_min, u_max, y_min, y_max

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "normal": [float(v) for v in self.normal],
            "d": float(self.d),
            "u": [float(v) for v in self.u],
            "extent": [float(v) for v in self.extent],
        }


@dataclass(frozen=True)
class Room:
    floor_y: float | None
    ceiling_y: float | None
    walls: list[Wall]

    def wall_distance(self, p: np.ndarray, wall: Wall) -> np.ndarray:
        """Signed distance in metres, positive on the room side of the wall."""
        return np.asarray(p, dtype=np.float64) @ wall.normal + wall.d

    def inside_xz(self, p: np.ndarray, margin_m: float) -> np.ndarray:
        """True for points not farther than margin_m behind any wall; all True with fewer than two walls."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        if len(self.walls) < 2:
            return np.ones(len(p), dtype=bool)
        ok = np.ones(len(p), dtype=bool)
        for w in self.walls:
            ok &= self.wall_distance(p, w) >= -margin_m
        return ok

    def to_json(self) -> dict:
        return {
            "floor_y": None if self.floor_y is None else float(self.floor_y),
            "ceiling_y": None if self.ceiling_y is None else float(self.ceiling_y),
            "walls": [w.to_json() for w in self.walls],
        }


def depth_point_cloud(depths, Ks_small: np.ndarray, c2w_cv: np.ndarray, voxel_m: float = 0.03, stride: int = 4) -> np.ndarray:
    """World points from rendered depth maps (every stride-th pixel), deduplicated on a voxel grid."""
    out = []
    for depth, K, c2w in zip(depths, Ks_small, c2w_cv):
        d = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
        vs, us = np.nonzero(d > 0)
        if len(vs) == 0:
            continue
        z = d[vs, us]
        us = us * stride + 0.5
        vs = vs * stride + 0.5
        x = (us - K[0, 2]) / K[0, 0] * z
        y = (vs - K[1, 2]) / K[1, 1] * z
        cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
        out.append((cam @ np.asarray(c2w, dtype=np.float64).T)[:, :3])
    if not out:
        return np.zeros((0, 3))
    pts = np.concatenate(out)
    keys = np.floor(pts / voxel_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(first)]


def _plane_from_points(p: np.ndarray) -> tuple[np.ndarray, float] | None:
    n = np.cross(p[1] - p[0], p[2] - p[0])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n = n / norm
    return n, float(-n @ p[0])


def _ransac_plane(points: np.ndarray, want: str, inlier_m: float, angle_deg: float, iters: int, rng) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Best plane of kind `want` ("horizontal": normal within angle_deg of ±Y; "vertical": at least
    angle_deg from Y) among `iters` random triplets. Returns (normal, d, inlier mask) or None."""
    if len(points) < 3:
        return None
    cos_lim = np.cos(np.radians(angle_deg))
    best = None
    for _ in range(iters):
        idx = rng.choice(len(points), 3, replace=False)
        fit = _plane_from_points(points[idx])
        if fit is None:
            continue
        n, d = fit
        c = abs(n @ UP)
        if want == "horizontal" and c < cos_lim:
            continue
        if want == "vertical" and c > cos_lim:
            continue
        dist = np.abs(points @ n + d)
        inl = dist < inlier_m
        count = int(inl.sum())
        if best is None or count > best[0]:
            best = (count, n, d, inl)
    if best is None:
        return None
    _, n, d, inl = best
    # refine on the inliers (least squares via SVD)
    q = points[inl]
    centroid = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - centroid, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    d = float(-n @ centroid)
    inl = np.abs(points @ n + d) < inlier_m
    return n, d, inl


def fit_room(
    points: np.ndarray,
    cam_positions: np.ndarray,
    *,
    min_floor_area_m2: float = 2.0,
    min_wall_area_m2: float = 1.5,
    inlier_m: float = DEFAULT_INLIER_M,
    horizontal_deg: float = 5.0,
    vertical_deg: float = 85.0,
    merge_deg: float = 10.0,
    merge_dist_m: float = 0.3,
    voxel_m: float = 0.03,
    iters: int = 300,
    seed: int = 0,
) -> Room:
    """Floor = lowest horizontal plane with enough area; ceiling = highest horizontal plane above every
    camera; walls = vertical planes by area, near-coplanar ones merged. Area = inliers x voxel_m^2."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=np.float64)
    cams = np.atleast_2d(np.asarray(cam_positions, dtype=np.float64))
    remaining = pts.copy()
    area_per_pt = voxel_m * voxel_m

    # --- horizontal planes ---
    horizontals: list[tuple[float, int]] = []  # (y, inlier count)
    while len(remaining) >= 3:
        fit = _ransac_plane(remaining, "horizontal", inlier_m, horizontal_deg, iters, rng)
        if fit is None:
            break
        n, d, inl = fit
        if inl.sum() * area_per_pt < min_floor_area_m2:
            break
        y = float(remaining[inl][:, 1].mean())
        horizontals.append((y, int(inl.sum())))
        remaining = remaining[~inl]
    floor_y = ceiling_y = None
    if horizontals:
        cam_min, cam_max = cams[:, 1].min(), cams[:, 1].max()
        below = [y for y, _ in horizontals if y < cam_min]
        above = [y for y, _ in horizontals if y > cam_max]
        floor_y = min(below) if below else None
        ceiling_y = max(above) if above else None

    # --- vertical planes ---
    walls: list[Wall] = []
    raw: list[tuple[np.ndarray, float, np.ndarray]] = []
    while len(remaining) >= 3:
        fit = _ransac_plane(remaining, "vertical", inlier_m, vertical_deg, iters, rng)
        if fit is None:
            break
        n, d, inl = fit
        if inl.sum() * area_per_pt < min_wall_area_m2:
            break
        raw.append((n, d, remaining[inl]))
        remaining = remaining[~inl]
    # merge near-coplanar planes
    merged: list[tuple[np.ndarray, float, np.ndarray]] = []
    cos_merge = np.cos(np.radians(merge_deg))
    for n, d, q in sorted(raw, key=lambda t: -len(t[2])):
        for i, (mn, md, mq) in enumerate(merged):
            if abs(n @ mn) < cos_merge:
                continue
            # orient both planes the same way before comparing offsets (opposite walls share |d|)
            dd = d if n @ mn > 0 else -d
            if abs(dd - md) < merge_dist_m:
                merged[i] = (mn, md, np.concatenate([mq, q]))
                break
        else:
            merged.append((n, d, q))
    centre = cams.mean(axis=0)
    for wid, (n, d, q) in enumerate(merged, start=1):
        n = n.copy()
        n[1] = 0.0
        n /= np.linalg.norm(n)
        d = float(-n @ q.mean(axis=0))
        if centre @ n + d < 0:  # make the normal point towards the cameras (into the room)
            n, d = -n, -d
        u = np.array([-n[2], 0.0, n[0]])
        proj_u = q @ u
        walls.append(Wall(wid, n, d, u, (float(proj_u.min()), float(proj_u.max()), float(q[:, 1].min()), float(q[:, 1].max()))))
    return Room(floor_y, ceiling_y, walls)
