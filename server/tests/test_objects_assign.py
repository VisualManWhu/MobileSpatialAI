import numpy as np
from PIL import Image

from recon3d.objects.assign import assign_by_voxels, assign_structure
from recon3d.objects.segments import VOXEL_M, Segment
from recon3d.objects.structure import Room, Wall
from recon3d.objects.thumbs import crop_thumbnail


def block(x0, y0, z0, n, frames=(0, 1, 2, 3, 4)):
    g = np.stack(np.mgrid[0:n, 0:n, 0:n], -1).reshape(-1, 3) + np.array([x0, y0, z0])
    lo = g.min(0) * VOXEL_M
    hi = (g.max(0) + 1) * VOXEL_M
    size = float(n * VOXEL_M)
    return Segment([1], list(frames), g, lo, hi, (size, size, size), {}, [])


def test_voxel_hits_assign_smallest_object_and_respect_the_one_voxel_halo():
    big = block(0, 0, 0, 20)  # 0.4 m cube
    small = block(5, 5, 5, 4)  # 8 cm cube inside it
    positions = np.array(
        [
            [0.11, 0.11, 0.11],  # inside small -> small (id 2)
            [0.35, 0.35, 0.35],  # inside big only -> big (id 1)
            [0.41, 0.2, 0.2],  # 1 cm past big's face: within the one-voxel (2 cm) halo -> big
            [0.47, 0.2, 0.2],  # 7 cm past: nothing
            [5.0, 5.0, 5.0],  # far away: nothing
        ]
    )
    ids = assign_by_voxels(positions, [big, small])
    assert ids.dtype == np.uint16
    assert ids.tolist() == [2, 1, 1, 0, 0]


def test_structure_assignment_fills_only_unassigned_gaussians_in_order():
    wall = Wall(1, np.array([1.0, 0.0, 0.0]), 2.0, np.array([0.0, 0.0, 1.0]), (-2.0, 2.0, 0.0, 2.6))  # x = -2
    room = Room(0.0, 2.6, [wall])
    door = block(-100, 0, 0, 5)  # a slab at x in [-2.0, -1.9], y 0..0.1, z 0..0.1 (stands in for a door's voxels)
    door.bbox_min = np.array([-2.1, 0.0, -0.5])
    door.bbox_max = np.array([-1.9, 2.0, 0.5])
    positions = np.array(
        [
            [0.0, 0.02, 0.0],  # on the floor
            [0.0, 2.58, 0.0],  # on the ceiling
            [-1.98, 1.5, 1.5],  # on the wall, outside the door's bbox
            [-1.98, 1.0, 0.0],  # on the wall, inside the door's bbox -> door, not wall
            [0.0, 1.3, 0.0],  # mid-air: nothing
            [0.0, 0.02, 0.5],  # on the floor but already assigned to object 7 -> untouched
        ]
    )
    ids = np.zeros(len(positions), dtype=np.uint16)
    ids[5] = 7
    out, created = assign_structure(positions, room, ids, first_id=10, openings=[(13, door)])
    # created: floor 10, ceiling 11, wall 12 (openings keep the id they were given)
    assert created == [10, 11, 12]
    assert out.tolist() == [10, 11, 12, 13, 0, 7]


def test_structure_assignment_without_floor_or_walls():
    positions = np.array([[0.0, 0.0, 0.0]])
    out, created = assign_structure(positions, Room(None, None, []), np.zeros(1, np.uint16), first_id=1, openings=[])
    assert created == [] and out.tolist() == [0]


def test_crop_thumbnail_pads_and_limits_long_side():
    img = Image.new("RGB", (1920, 1080))
    t = crop_thumbnail(img, (900, 500, 1100, 600))  # 200x100 box -> padded 260x130 -> long side 256
    assert max(t.size) == 256 and abs(t.size[0] / t.size[1] - 2.0) < 0.05
    edge = crop_thumbnail(img, (1800, 1000, 1920, 1080))  # clipped at the image border, still an image
    assert edge.size[0] > 0 and edge.size[1] > 0
