import numpy as np
import pytest

from recon3d.objects.merge import Detection, bbox_iou, merge_detections, nms_frame, point_overlap


def cloud(center, size=0.2, n=200, seed=0):
    r = np.random.default_rng(seed)
    return np.asarray(center, dtype=float) + r.uniform(-size / 2, size / 2, (n, 3))


def det(frame, label, center, score=0.5, seed=0, box=(0, 0, 10, 10)):
    return Detection(frame, label, score, box, cloud(center, seed=seed))


def test_nms_drops_overlapping_lower_score_box_across_classes():
    a = det(0, "monitor", (0, 0, 0), 0.6, box=(0, 0, 100, 100))
    b = det(0, "tv", (0, 0, 0), 0.5, box=(5, 5, 100, 100))
    c = det(0, "cup", (1, 0, 0), 0.9, box=(200, 200, 220, 220))
    kept = nms_frame([a, b, c])
    assert sorted(d.label for d in kept) == ["cup", "monitor"]
    assert all(d is not b for d in kept)


def test_point_overlap_and_bbox_iou():
    a = cloud((0, 0, 0))
    b = cloud((0, 0, 0), seed=1)
    far = cloud((2, 0, 0))
    assert point_overlap(a, b, 0.05) > 0.9 and point_overlap(a, far, 0.05) == 0.0
    assert bbox_iou(np.zeros(3), np.ones(3), np.zeros(3), np.ones(3)) == 1.0
    assert bbox_iou(np.zeros(3), np.ones(3), np.array([0.5, 0, 0]), np.array([1.5, 1, 1])) == pytest.approx(1 / 3)


def test_same_object_seen_in_three_frames_merges_to_one():
    dets = [det(f, "chair", (1, 0.5, -2), seed=f) for f in range(3)]
    objs = merge_detections(dets)
    assert len(objs) == 1 and objs[0].views == 3 and sorted(objs[0].members) == [0, 1, 2]
    assert np.allclose(objs[0].center, (1, 0.5, -2), atol=0.03)


def test_two_touching_cups_in_the_same_frame_stay_two_objects():
    # same frame -> never merged, even though the clouds overlap heavily
    dets = [
        det(0, "cup", (0, 0, 0), seed=0),
        det(0, "cup", (0.05, 0, 0), seed=1),
        det(1, "cup", (0, 0, 0), seed=2),
        det(1, "cup", (0.05, 0, 0), seed=3),
    ]
    objs = merge_detections(dets)
    assert len(objs) == 2 and all(o.views == 2 for o in objs)


def test_single_view_object_is_dropped():
    dets = [det(0, "cup", (0, 0, 0)), det(1, "chair", (3, 0, 0)), det(2, "chair", (3, 0, 0), seed=1)]
    objs = merge_detections(dets)
    assert [o.label for o in objs] == ["chair"]


def test_cross_class_merge_needs_strong_overlap_and_votes_label_by_score():
    # monitor/tv flicker on the same object: 2 monitor views at 0.5, 1 tv view at 0.9 -> label by score sum
    dets = [
        det(0, "monitor", (0, 1, -1), 0.5),
        det(1, "tv", (0, 1, -1), 0.9, seed=1),
        det(2, "monitor", (0, 1, -1), 0.5, seed=2),
    ]
    objs = merge_detections(dets)
    assert len(objs) == 1 and objs[0].label == "monitor" and objs[0].views == 3
    # weak overlap across classes does not merge (clouds offset 15 cm, size 20 cm)
    dets2 = [
        det(0, "bag", (0, 0, 0)),
        det(1, "backpack", (0.15, 0, 0), seed=1),
        det(2, "bag", (0, 0, 0), seed=2),
        det(3, "backpack", (0.15, 0, 0), seed=3),
    ]
    assert len(merge_detections(dets2)) == 2


def test_bbox_is_5_95_percentile_and_score_is_max():
    pts = np.array([[0, 0, 0]] * 98 + [[10, 10, 10], [-10, -10, -10]], dtype=float)
    dets = [Detection(0, "box", 0.4, (0, 0, 1, 1), pts), Detection(1, "box", 0.7, (0, 0, 1, 1), pts)]
    o = merge_detections(dets)[0]
    assert np.allclose(o.bbox_min, 0) and np.allclose(o.bbox_max, 0) and o.score == 0.7
