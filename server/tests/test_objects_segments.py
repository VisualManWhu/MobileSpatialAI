import numpy as np

from recon3d.objects.segments import (
    MIN_TRACK_FRAMES,
    Segment,
    Track,
    merge_segments,
    support_of,
    track_to_segment,
    voxel_key,
)
from recon3d.objects.structure import Room, Wall

H, W = 216, 384
K = np.array([[320.0, 0, W / 2], [0, 320.0, H / 2], [0, 0, 1]])


def look_at(eye, target):
    """OpenCV camera-to-world (+Z forward, +Y down) looking from eye at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(-up, z)  # OpenCV y is down
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def render_box(c2w, lo, hi):
    """Analytic depth of an axis-aligned box [lo, hi] (and nothing else): per-pixel ray/AABB test."""
    vs, us = np.mgrid[0:H, 0:W]
    dirs_cam = np.stack([(us + 0.5 - K[0, 2]) / K[0, 0], (vs + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(us, float)], -1)
    dirs = dirs_cam @ c2w[:3, :3].T
    eye = c2w[:3, 3]
    t0 = (lo - eye) / dirs
    t1 = (hi - eye) / dirs
    tmin = np.max(np.minimum(t0, t1), axis=-1)
    tmax = np.min(np.maximum(t0, t1), axis=-1)
    hit = (tmax >= tmin) & (tmin > 0)
    depth = np.where(hit, tmin * dirs_cam[..., 2], 0.0).astype(np.float32)
    return depth, hit


LO, HI = np.array([-0.2, 0.0, -0.2]), np.array([0.2, 0.4, 0.2])
EYES = [[0.0, 0.8, 2.0], [1.6, 0.9, 1.2], [-1.6, 0.9, 1.2], [0.0, 1.2, -2.0], [1.4, 0.5, -1.4], [-1.4, 0.5, -1.4]]


def box_frames(n=6, lo=LO, hi=HI):
    c2w = np.stack([look_at(e, (hi + lo) / 2) for e in EYES[:n]])
    depths, masks = [], []
    for c in c2w:
        d, hit = render_box(c, lo, hi)
        depths.append(d)
        masks.append(np.packbits(hit))
    return c2w, depths, masks


def track(frames, masks, boxes=None, tid=1, votes=None):
    boxes = boxes or [(0.0, 0.0, 10.0, 10.0)] * len(frames)
    return Track(tid, list(frames), list(masks), boxes, votes or {"box": 1.0})


def test_track_to_segment_keeps_voxels_hit_in_three_frames():
    c2w, depths, masks = box_frames()
    Ks = np.stack([K] * 6)
    seg = track_to_segment(track(range(6), masks), depths, Ks, c2w, H, W)
    assert seg is not None
    l, w, h = seg.size_lwh
    assert abs(l - 0.4) < 0.06 and abs(w - 0.4) < 0.06 and abs(h - 0.4) < 0.06
    assert l >= w
    assert np.all(seg.bbox_min > LO - 0.05) and np.all(seg.bbox_max < HI + 0.05)
    assert seg.frames == list(range(6)) and seg.track_ids == [1]
    assert 1 <= len(seg.best_views) <= 5 and all(f in range(6) for f, _ in seg.best_views)
    # a frame whose mask covers only empty background adds no voxels
    bg = np.packbits(np.ones((H, W), bool))
    depth_bg = np.zeros((H, W), np.float32)
    seg2 = track_to_segment(
        track(range(7), masks + [bg]), depths + [depth_bg], np.stack([K] * 7), np.concatenate([c2w, c2w[:1]]), H, W
    )
    assert len(seg2.voxels) == len(seg.voxels)


def test_track_without_depth_is_dropped():
    c2w, _, masks = box_frames()
    zero = [np.zeros((H, W), np.float32)] * 6
    assert track_to_segment(track(range(6), masks), zero, np.stack([K] * 6), c2w, H, W) is None


def test_short_track_is_dropped():
    c2w, depths, masks = box_frames()
    n = MIN_TRACK_FRAMES - 1
    assert track_to_segment(track(range(n), masks[:n]), depths[:n], np.stack([K] * n), c2w[:n], H, W) is None


def _seg_from(frames, tid, votes=None):
    c2w, depths, masks = box_frames()
    Ks = np.stack([K] * 6)
    s = track_to_segment(track(range(6), masks, tid=tid, votes=votes), depths, Ks, c2w, H, W)
    s.frames = list(frames)
    return s


def test_merge_requires_no_shared_frame_and_half_overlap():
    a = _seg_from([0, 1, 2, 3, 4], 1, {"box": 2.0})
    b = _seg_from([10, 11, 12, 13, 14], 2, {"box": 1.0, "cup": 0.5})
    merged = merge_segments([a, b])
    assert len(merged) == 1
    m = merged[0]
    assert m.frames == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14] and sorted(m.track_ids) == [1, 2]
    assert m.votes == {"box": 3.0, "cup": 0.5}
    # one shared frame -> stays apart
    c = _seg_from([4, 20, 21, 22, 23], 3)
    assert len(merge_segments([a, c])) == 2
    # disjoint frames but little overlap -> apart
    d = _seg_from([30, 31, 32, 33, 34], 4)
    d.voxels = d.voxels + np.array([40, 0, 0])  # shifted 0.8 m
    assert len(merge_segments([a, d])) == 2


def _plain(voxels, frames=(0, 1, 2, 3, 4)):
    v = np.asarray(voxels, dtype=np.int64)
    centres = (v + 0.5) * 0.02
    lo, hi = centres.min(0), centres.max(0)
    return Segment([9], list(frames), v, lo, hi, (float(hi[0] - lo[0]), float(hi[2] - lo[2]), float(hi[1] - lo[1])), {}, [])


def cube(x0, y0, z0, n):
    g = np.stack(np.mgrid[0:n, 0:n, 0:n], -1).reshape(-1, 3)
    return g + np.array([x0, y0, z0])


def test_support_floor_on_wall_none():
    wall = Wall(1, np.array([1.0, 0.0, 0.0]), 2.0, np.array([0.0, 0.0, 1.0]), (-2.0, 2.0, 0.0, 2.6))  # x = -2
    room = Room(0.0, 2.6, [wall])
    table = _plain(cube(0, 0, 0, 40))  # 0.8 m cube on the floor
    assert support_of(table, room, []) == "floor"
    cup = _plain(cube(10, 42, 10, 5))  # bottom at y = 0.84, table top at 0.80 (+0.04)
    assert support_of(cup, room, [table]) == "on:0"
    poster = _plain(cube(-99, 50, 0, 3))  # x voxels -99..-97 -> within 15 cm of the x = -2 wall
    assert support_of(poster, room, [table]) == "wall:1"
    floating = _plain(cube(60, 60, 60, 5))
    assert support_of(floating, room, [table]) == "none"
    assert support_of(table, Room(None, None, []), []) == "none"


def test_voxel_key_floors():
    k = voxel_key(np.array([[0.031, -0.001, 0.0]]))
    assert k.tolist() == [[1, -1, 0]]
