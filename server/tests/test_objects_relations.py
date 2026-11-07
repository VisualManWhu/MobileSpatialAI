import numpy as np

from recon3d.objects.relations import compute_relations
from recon3d.objects.table import ObjectRecord


def obj(i, label, mn, mx):
    c = tuple((np.asarray(mn, dtype=float) + np.asarray(mx, dtype=float)) / 2)
    return ObjectRecord(i, label, label, 0.5, 2, c, tuple(mn), tuple(mx), 10)


table = obj(1, "table", (-0.6, 0.0, -0.4), (0.6, 0.75, 0.4))
cup = obj(2, "cup", (0.1, 0.75, 0.0), (0.2, 0.85, 0.1))
book = obj(3, "book", (-0.3, 0.76, 0.0), (-0.1, 0.78, 0.15))
chair = obj(4, "chair", (0.75, 0.0, -0.3), (1.2, 0.9, 0.2))
far = obj(5, "door", (5, 0, 0), (5.1, 2, 0.9))
box = obj(6, "box", (-2.0, 0.0, -2.0), (-1.0, 1.0, -1.0))
pen = obj(7, "pen", (-1.6, 0.4, -1.6), (-1.5, 0.42, -1.4))


def rels(*objs, look=(0, 0, -1)):
    return compute_relations(list(objs), np.array(look, dtype=float))


def test_on_and_in():
    r = rels(table, cup, box, pen)
    assert {"a": 2, "rel": "on", "b": 1} in r
    assert {"a": 7, "rel": "in", "b": 6} in r
    assert {"a": 1, "rel": "on", "b": 2} not in r


def test_next_to_is_unordered_and_needs_small_gap():
    r = rels(table, chair, far)
    assert {"a": 1, "rel": "next_to", "b": 4} in r and {"a": 4, "rel": "next_to", "b": 1} not in r
    assert not any(x["b"] == 5 or x["a"] == 5 for x in r)


def test_left_right_from_home_view():
    # looking down -Z, +X is to the right: cup (x=.15) is right of book (x=-.2)
    r = rels(cup, book)
    assert {"a": 2, "rel": "right_of", "b": 3} in r and {"a": 3, "rel": "left_of", "b": 2} in r
    assert all(x["rel"] not in ("left_of", "right_of") for x in compute_relations([cup, book], None))


def test_far_pairs_produce_nothing():
    assert compute_relations([table, far], np.array([0, 0, -1.0])) == []
