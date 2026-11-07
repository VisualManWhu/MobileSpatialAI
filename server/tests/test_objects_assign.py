import numpy as np
from PIL import Image

from recon3d.objects.assign import ObjectMasks, assign_gaussians
from recon3d.objects.thumbs import crop_thumbnail

W, H = 64, 48
K = np.array([[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]])
cam = np.eye(4)  # OpenCV camera at origin looking +Z


def frame_with_object(frame, depth_m, mask_box, obj_id=1):
    depth = np.full((H, W), depth_m, dtype=np.float32)
    mask = np.zeros((H, W), bool)
    x0, y0, x1, y1 = mask_box
    mask[y0:y1, x0:x1] = True
    return depth, ObjectMasks(frame, obj_id, mask)


def test_gaussian_inside_mask_and_at_depth_in_two_frames_gets_the_object():
    pos = np.array(
        [
            [0.0, 0.0, 2.0],  # projects to (32,24), depth 2 -> inside mask, right depth
            [0.0, 0.0, 3.0],  # same pixel, 1 m behind the surface -> depth mismatch
            [1.5, 0.0, 2.0],  # projects to x=77 -> off image
        ]
    )
    d0, m0 = frame_with_object(0, 2.0, (20, 12, 44, 36))
    d1, m1 = frame_with_object(1, 2.0, (20, 12, 44, 36))
    ids = assign_gaussians(
        pos, np.stack([cam, cam]), np.stack([K, K]), [d0, d1], [m0, m1],
        bbox_min=np.array([[-1, -1, 1.5]]), bbox_max=np.array([[1, 1, 2.5]]),
    )
    assert ids.dtype == np.uint16 and ids.tolist() == [1, 0, 0]


def test_one_vote_is_not_enough_and_bbox_margin_is_enforced():
    pos = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    d0, m0 = frame_with_object(0, 2.0, (0, 0, W, H))
    ids = assign_gaussians(pos, cam[None], K[None], [d0], [m0], np.array([[-1, -1, 1.5]]), np.array([[1, 1, 2.5]]))
    assert ids.tolist() == [0, 0]
    d1, m1 = frame_with_object(1, 2.0, (0, 0, W, H))
    tight = assign_gaussians(
        pos, np.stack([cam, cam]), np.stack([K, K]), [d0, d1], [m0, m1],
        np.array([[0.5, 0.5, 1.5]]), np.array([[1, 1, 2.5]]),  # object box excludes x=0 (margin 0.1)
    )
    assert tight.tolist() == [0, 0]


def test_votes_pick_the_object_with_most_views():
    pos = np.array([[0.0, 0.0, 2.0]])
    depths, masks = [], []
    for f in range(5):
        depths.append(np.full((H, W), 2.0, np.float32))
        masks.append(ObjectMasks(f, 2 if f < 3 else 1, np.ones((H, W), bool)))
    ids = assign_gaussians(
        pos, np.stack([cam] * 5), np.stack([K] * 5), depths, masks,
        np.array([[-1, -1, 1.5]] * 2), np.array([[1, 1, 2.5]] * 2),
    )
    assert ids.tolist() == [2]


def test_crop_thumbnail_pads_and_limits_long_side():
    img = Image.new("RGB", (1920, 1080))
    t = crop_thumbnail(img, (900, 500, 1100, 600))  # 200x100 box -> padded 260x130 -> long side 256
    assert max(t.size) == 256 and abs(t.size[0] / t.size[1] - 2.0) < 0.05
    edge = crop_thumbnail(img, (1800, 1000, 1920, 1080))  # clipped at the image border, still an image
    assert edge.size[0] > 0 and edge.size[1] > 0
