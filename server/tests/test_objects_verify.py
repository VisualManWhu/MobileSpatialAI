import random

import numpy as np
from PIL import Image

from recon3d.objects import vocab
from recon3d.objects.segments import VOXEL_M, Segment
from recon3d.objects.structure import Room, Wall
from recon3d.objects.verify import (
    Verified,
    candidate_entries,
    confidence_of,
    confirm,
    door_window_ok,
    harmony_filter,
    parse_answer,
)


class FakeVlm:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.image_counts: list[int] = []

    def ask(self, images, prompt):
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        return self.answers.pop(0)


def seg(size_lwh, bottom_y=0.0, votes=None, frames=(0, 1, 2, 3, 4), x=0.0, z=0.0, fill=True):
    """Hand-built segment: a solid (or hollow) block of voxels of the given size at (x, bottom_y, z)."""
    l, w, h = size_lwh
    nx, ny, nz = max(1, round(l / VOXEL_M)), max(1, round(h / VOXEL_M)), max(1, round(w / VOXEL_M))
    g = np.stack(np.mgrid[0:nx, 0:ny, 0:nz], -1).reshape(-1, 3)
    if not fill:  # keep only the outer shell so a thin thing stays thin
        g = g[(g[:, 0] == 0) | (g[:, 0] == nx - 1) | (g[:, 2] == 0) | (g[:, 2] == nz - 1) | (g[:, 1] == 0) | (g[:, 1] == ny - 1)]
    origin = np.array([x - l / 2, bottom_y, z - w / 2])
    voxels = g + np.floor(origin / VOXEL_M).astype(np.int64)
    lo = origin
    hi = origin + np.array([l, h, w])
    return Segment([1], list(frames), voxels, lo, hi, (max(l, w), min(l, w), h), dict(votes or {}), [(0, (0.0, 0.0, 10.0, 10.0))])


WALL_X0 = Wall(1, np.array([1.0, 0.0, 0.0]), 2.0, np.array([0.0, 0.0, 1.0]), (-2.5, 2.5, 0.0, 2.6))  # x = -2
WALL_X1 = Wall(2, np.array([-1.0, 0.0, 0.0]), 2.0, np.array([0.0, 0.0, 1.0]), (-2.5, 2.5, 0.0, 2.6))  # x = +2
ROOM = Room(0.0, 2.6, [WALL_X0, WALL_X1])
IMG = [Image.new("RGB", (64, 64), "white")] * 3


def test_candidates_ordered_by_votes_then_truncated():
    s = seg((0.15, 0.07, 0.01), bottom_y=0.75, votes={"remote control": 0.9, "phone": 2.5})
    names = [e.en for e in candidate_entries(s, "on:3", "item")]
    assert names[:2] == ["phone", "remote control"]
    assert len(names) <= 5
    assert "monitor" not in names


def test_parse_answer_accepts_wrapped_name_rejects_extra_text():
    opts = [vocab.entry("door"), vocab.entry("cabinet")]
    assert parse_answer("「门」", opts).en == "door"
    assert parse_answer(" 门。", opts).en == "door"
    assert parse_answer("这是门。", opts) is None
    assert parse_answer("不确定", opts) is None
    assert parse_answer("", opts) is None


def test_confirm_needs_two_matching_answers():
    opts = [vocab.entry("door"), vocab.entry("cabinet"), vocab.entry("window")]
    s = seg((0.9, 0.1, 2.0))
    vlm = FakeVlm(["门", "门"])
    assert confirm(s, "wall:1", opts, IMG, vlm, random.Random(1)).en == "door"
    assert len(vlm.prompts) == 2 and vlm.image_counts == [3, 3]
    assert "2.00 米" in vlm.prompts[0] and "0.90 米" in vlm.prompts[0] and "贴墙" in vlm.prompts[0]
    # the second ask lists the options in a different order (shuffled) for at least one of several seeds
    differs = []
    for sd in range(5):
        v = FakeVlm(["门", "门"])
        confirm(s, "wall:1", opts, IMG, v, random.Random(sd))
        differs.append(v.prompts[0] != v.prompts[1])
    assert any(differs)
    assert confirm(s, "wall:1", opts, IMG, FakeVlm(["门", "柜子"]), random.Random(1)) is None
    assert confirm(s, "wall:1", opts, IMG, FakeVlm(["不确定", "门"]), random.Random(1)) is None


def test_white_door_never_becomes_cabinet():
    door = seg((0.1, 0.9, 2.0), bottom_y=0.0, x=-1.95, fill=False)  # flush with the x = -2 wall, on the floor
    assert door_window_ok(door, vocab.entry("door"), ROOM)
    assert not door_window_ok(door, vocab.entry("window"), ROOM)  # bottom on the floor -> not a window
    high = seg((0.1, 0.9, 1.0), bottom_y=0.9, x=-1.95, fill=False)
    assert door_window_ok(high, vocab.entry("window"), ROOM) and not door_window_ok(high, vocab.entry("door"), ROOM)
    away = seg((0.1, 0.9, 2.0), bottom_y=0.0, x=0.0, fill=False)
    assert not door_window_ok(away, vocab.entry("door"), ROOM)
    # labelled cabinet by the VLM: the harmony filter removes a tall wall-flush slab
    v = Verified(door, vocab.entry("cabinet"), "wall:1", 0.8)
    assert harmony_filter([v], ROOM) == []
    # a real cabinet standing 40 cm proud of the wall stays
    cab = seg((0.4, 0.9, 2.0), bottom_y=0.0, x=-1.75)
    assert harmony_filter([Verified(cab, vocab.entry("cabinet"), "floor", 0.8)], ROOM) != []


def test_furniture_overlap_keeps_larger():
    big = seg((1.2, 0.6, 0.75), votes={"table": 1})
    small = seg((0.8, 0.5, 0.7), x=0.1, votes={"desk": 1})
    kept = harmony_filter([Verified(small, vocab.entry("desk"), "floor", 0.5), Verified(big, vocab.entry("table"), "floor", 0.5)], ROOM)
    assert [k.entry.en for k in kept] == ["table"]


def test_item_inside_furniture_dropped_but_object_on_top_kept():
    table = seg((1.2, 0.6, 0.75))
    inside = seg((0.3, 0.3, 0.3), bottom_y=0.2)  # a chunk of the table itself
    cup = seg((0.08, 0.08, 0.1), bottom_y=0.75, x=0.2)
    kept = harmony_filter(
        [
            Verified(table, vocab.entry("table"), "floor", 0.9),
            Verified(inside, vocab.entry("box"), "none", 0.9),
            Verified(cup, vocab.entry("cup"), "on:0", 0.9),
        ],
        ROOM,
    )
    assert [k.entry.en for k in kept] == ["table", "cup"]


def test_outside_walls_dropped_only_with_two_walls():
    out = seg((0.4, 0.4, 0.4), x=2.6)  # 0.6 m beyond the x = +2 wall
    v = [Verified(out, vocab.entry("box"), "floor", 0.9)]
    assert harmony_filter(v, ROOM) == []
    assert harmony_filter(v, Room(0.0, None, [WALL_X0])) == v


def test_confidence_is_vote_share():
    s = seg((0.4, 0.4, 0.9), votes={"chair": 3.0, "stool": 1.0})
    assert abs(confidence_of(s, vocab.entry("chair")) - 0.75) < 1e-9
    assert confidence_of(seg((0.4, 0.4, 0.9)), vocab.entry("chair")) == 0.0
