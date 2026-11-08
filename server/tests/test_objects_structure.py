import numpy as np

from recon3d.objects import structure
from recon3d.objects.structure import Room, Wall, depth_point_cloud, fit_room

# A 4 x 5 m room, floor y = 0, ceiling y = 2.6; cameras at y = 1.4 inside.
X0, X1, Z0, Z1, CEIL = -2.0, 2.0, -2.5, 2.5, 2.6


def _grid(a0, a1, b0, b1, step=0.05):
    a, b = np.meshgrid(np.arange(a0, a1, step), np.arange(b0, b1, step), indexing="ij")
    return a.ravel(), b.ravel()


def box_room(floor=True, ceiling=True, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    x, z = _grid(X0, X1, Z0, Z1)
    if floor:
        parts.append(np.stack([x, np.zeros_like(x), z], 1))
    if ceiling:
        parts.append(np.stack([x, np.full_like(x, CEIL), z], 1))
    x, y = _grid(X0, X1, 0.0, CEIL)
    parts.append(np.stack([x, y, np.full_like(x, Z0)], 1))
    parts.append(np.stack([x, y, np.full_like(x, Z1)], 1))
    z, y = _grid(Z0, Z1, 0.0, CEIL)
    parts.append(np.stack([np.full_like(z, X0), y, z], 1))
    parts.append(np.stack([np.full_like(z, X1), y, z], 1))
    pts = np.concatenate(parts)
    pts += rng.normal(0, 0.005, pts.shape)
    noise = rng.uniform([X0 + 0.3, 0.3, Z0 + 0.3], [X1 - 0.3, CEIL - 0.3, Z1 - 0.3], (500, 3))
    return np.concatenate([pts, noise])


CAMS = np.array([[0.0, 1.4, 0.0], [1.0, 1.4, -1.0], [-1.0, 1.4, 1.0]])


def test_fit_room_finds_floor_ceiling_and_four_walls():
    room = fit_room(box_room(), CAMS)
    assert abs(room.floor_y) < 0.02
    assert abs(room.ceiling_y - CEIL) < 0.02
    assert len(room.walls) == 4
    centre = np.array([0.0, 1.3, 0.0])
    for w in room.walls:
        assert abs(np.linalg.norm(w.normal) - 1) < 1e-6 and abs(w.normal[1]) < 0.1
        # the normal points into the room: moving from the wall towards the centre goes along it
        on_wall = -w.d * w.normal
        assert np.dot(centre - on_wall, w.normal) > 0
        u0, u1, y0, y1 = w.extent
        assert u1 - u0 > 3.5 and y0 < 0.1 and y1 > CEIL - 0.1


def test_no_floor_in_view_yields_none():
    room = fit_room(box_room(floor=False), CAMS)
    assert room.floor_y is None and len(room.walls) == 4 and room.ceiling_y is not None


def test_inside_xz_margin():
    room = fit_room(box_room(), CAMS)
    p = np.array([[0.0, 1.0, 0.0], [X1 + 0.5, 1.0, 0.0], [X1 + 0.2, 1.0, 0.0]])
    inside = room.inside_xz(p, margin_m=0.3)
    assert inside.tolist() == [True, False, True]
    assert Room(None, None, []).inside_xz(p, 0.3).tolist() == [True, True, True]
    assert room.to_json()["floor_y"] is not None and len(room.to_json()["walls"]) == 4


def test_wall_distance_is_signed_metres():
    room = fit_room(box_room(), CAMS)
    w = max(room.walls, key=lambda w: w.normal[0])  # the x = X0 wall, normal +x
    d = room.wall_distance(np.array([[X0 + 0.5, 1.0, 0.0], [X0 - 0.2, 1.0, 0.0]]), w)
    assert abs(d[0] - 0.5) < 0.05 and abs(d[1] + 0.2) < 0.05


def test_depth_point_cloud_backprojects():
    h, w = 54, 96
    depth = np.full((h, w), 2.0, dtype=np.float32)
    depth[:5] = 0  # no surface on the top rows
    K = np.array([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    pts = depth_point_cloud([depth], np.array([K]), np.array([c2w]), voxel_m=0.03, stride=1)
    assert pts.shape[1] == 3 and 0 < len(pts) < h * w
    assert np.allclose(pts[:, 2], 3.0 + 2.0, atol=0.03)  # camera +Z = depth, translated by c2w
    coarse = depth_point_cloud([depth], np.array([K]), np.array([c2w]), voxel_m=0.3, stride=1)
    assert len(coarse) < len(pts)


def test_module_constants():
    assert structure.DEFAULT_INLIER_M == 0.03


def test_vertical_fit_never_turns_into_a_horizontal_plane():
    """A 1.6 m² table top sits below the floor-area threshold; the wall RANSAC must not adopt it after
    refinement (review finding #5): every wall normal stays horizontal."""
    pts = box_room()
    x, z = _grid(-0.6, 0.6, -0.7, 0.7, step=0.03)  # 1.68 m² at y = 0.75
    pts = np.concatenate([pts, np.stack([x, np.full_like(x, 0.75), z], 1)])
    room = fit_room(pts, CAMS)
    assert abs(room.floor_y) < 0.02
    for w in room.walls:
        assert abs(w.normal[1]) < 0.05


def test_inside_xz_only_counts_walls_the_point_faces():
    """An L-shaped room: a wall's infinite plane must not condemn points beyond the wall's own extent
    (review finding #6)."""
    short = Wall(1, np.array([1.0, 0.0, 0.0]), 0.0, np.array([0.0, 0.0, 1.0]), (0.0, 2.0, 0.0, 2.6))  # x = 0, only for z in [0, 2]
    far = Wall(2, np.array([-1.0, 0.0, 0.0]), 4.0, np.array([0.0, 0.0, 1.0]), (-3.0, 3.0, 0.0, 2.6))  # x = 4
    room = Room(0.0, None, [short, far])
    p = np.array([[-1.0, 1.0, 1.0], [-1.0, 1.0, -2.0], [5.0, 1.0, 0.0]])
    # behind the short wall within its extent -> out; behind its plane but past its end -> in; behind the far wall -> out
    assert room.inside_xz(p, 0.3).tolist() == [False, True, False]


def test_depth_point_cloud_is_float32_and_dedups_across_frames():
    h, w = 54, 96
    depth = np.full((h, w), 2.0, dtype=np.float32)
    K = np.array([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1]])
    one = depth_point_cloud([depth], np.array([K]), np.array([np.eye(4)]), voxel_m=0.03, stride=1)
    two = depth_point_cloud([depth, depth], np.array([K, K]), np.array([np.eye(4), np.eye(4)]), voxel_m=0.03, stride=1)
    assert one.dtype == np.float32 and len(two) == len(one)  # the same frame twice adds no points
