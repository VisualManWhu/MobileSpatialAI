"""Verification layer: size/support candidates, VLM two-ask confirmation and the harmony filter that
keeps the table trustworthy (spec 2026-11-08 §4.3). Pure Python apart from the injected VLM."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from recon3d.objects import vocab
from recon3d.objects.segments import VOXEL_M, Segment
from recon3d.objects.structure import Room
from recon3d.objects.vocab import Level, VocabEntry

TOP_K = 5
UNSURE = "不确定"
PROMPT = (
    "这是同一个室内物体的几张照片。它高约 {h:.2f} 米、最长边约 {l:.2f} 米，{where}。"
    "它是下列哪一个：{options}？只回答其中一个名称；都不是或看不清就回答「不确定」。"
)
WHERE = {"floor": "放在地面上", "on": "放在家具上", "wall": "贴墙", "ceiling": "装在天花板上", "none": "位置不明"}
_STRIP = " \t\r\n「」『』\"'“”‘’。.!！,，、:：;；"

# harmony rules
FURNITURE_IOU = 0.3
ITEM_IN_FURNITURE_FRACTION = 0.5  # of the item's bbox volume inside a furniture bbox (containment, not IoU)
OUTSIDE_MARGIN_M = 0.3
WALL_BAND_M = 0.15
WALL_SLAB_FRACTION = 0.8
WALL_SLAB_MIN_H = 1.5
DOOR_FLOOR_GAP_M = 0.15
WINDOW_MIN_SILL_M = 0.3


class Vlm(Protocol):
    def ask(self, images: list[Image.Image], prompt: str) -> str: ...


@dataclass
class Verified:
    seg: Segment
    entry: VocabEntry
    support: str
    confidence: float


def candidate_entries(seg: Segment, support: str, level: Level, top_k: int = TOP_K) -> list[VocabEntry]:
    """Vocab entries that pass the size/support gate, voted ones first (by summed proposer score)."""
    fits = vocab.candidates(seg.size_lwh, support, level)
    fits.sort(key=lambda e: -seg.votes.get(e.en, 0.0))
    return fits[:top_k]


def parse_answer(text: str, options: list[VocabEntry]) -> VocabEntry | None:
    """Exact match on one option's Chinese name after stripping quotes/punctuation; anything else is None."""
    name = (text or "").strip(_STRIP)
    for e in options:
        if name == e.zh:
            return e
    return None


def build_prompt(seg: Segment, support: str, options: list[VocabEntry]) -> str:
    l, w, h = seg.size_lwh
    return PROMPT.format(h=h, l=max(l, w), where=WHERE[vocab.support_kind(support)], options="、".join(e.zh for e in options))


def confirm(seg: Segment, support: str, options: list[VocabEntry], crops: list[Image.Image], vlm: Vlm, rng: random.Random) -> VocabEntry | None:
    """Ask twice (second time with the options shuffled); both answers must name the same option."""
    if not options:
        return None
    first = parse_answer(vlm.ask(crops, build_prompt(seg, support, options)), options)
    if first is None:
        return None
    shuffled = list(options)
    rng.shuffle(shuffled)
    second = parse_answer(vlm.ask(crops, build_prompt(seg, support, shuffled)), shuffled)
    return first if second is not None and second.en == first.en else None


def confidence_of(seg: Segment, entry: VocabEntry) -> float:
    total = sum(seg.votes.values())
    return float(seg.votes.get(entry.en, 0.0) / total) if total > 0 else 0.0


def _wall_fraction(seg: Segment, room: Room) -> float:
    """Largest fraction of the segment's voxels within WALL_BAND_M of any single wall plane."""
    if not room.walls:
        return 0.0
    centres = (seg.voxels + 0.5) * VOXEL_M
    return max(float((np.abs(room.wall_distance(centres, w)) < WALL_BAND_M).mean()) for w in room.walls)


def door_window_ok(seg: Segment, entry: VocabEntry, room: Room) -> bool:
    """Geometric gate for the structure classes: in a wall, door on the floor, window off the floor."""
    if _wall_fraction(seg, room) < WALL_SLAB_FRACTION:
        return False
    l, w, h = seg.size_lwh
    bottom = float(seg.bbox_min[1])
    if entry.en == "door":
        if room.floor_y is None or bottom - room.floor_y >= DOOR_FLOOR_GAP_M:
            return False
        return 1.8 <= h <= 2.4 and 0.6 <= max(l, w) <= 1.2
    if entry.en == "window":
        if room.floor_y is not None and bottom - room.floor_y < WINDOW_MIN_SILL_M:
            return False
        return 0.4 <= h <= 2.2
    return False


def bbox_iou(a: Segment, b: Segment) -> float:
    lo = np.maximum(a.bbox_min, b.bbox_min)
    hi = np.minimum(a.bbox_max, b.bbox_max)
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    va = float(np.prod(a.bbox_max - a.bbox_min))
    vb = float(np.prod(b.bbox_max - b.bbox_min))
    union = va + vb - inter
    return inter / union if union > 0 else 0.0


def inside_fraction(a: Segment, b: Segment) -> float:
    """Fraction of a's bbox volume that lies inside b's bbox."""
    lo = np.maximum(a.bbox_min, b.bbox_min)
    hi = np.minimum(a.bbox_max, b.bbox_max)
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    va = float(np.prod(a.bbox_max - a.bbox_min))
    return inter / va if va > 0 else 0.0


def harmony_filter(verified: list[Verified], room: Room) -> list[Verified]:
    """Spec §4.3 last gate, applied in order: furniture IoU > 0.3 keeps the larger; an item mostly
    inside a furniture bbox is a part of it; nothing lives outside the walls (two or more walls);
    a tall wall-flush slab is a door/window or nothing."""
    keep = list(verified)
    # 1. furniture overlap
    furniture = sorted([v for v in keep if v.entry.level == "furniture"], key=lambda v: -v.seg.volume())
    dropped: set[int] = set()
    for i, a in enumerate(furniture):
        if id(a) in dropped:
            continue
        for b in furniture[i + 1 :]:
            if id(b) not in dropped and bbox_iou(a.seg, b.seg) > FURNITURE_IOU:
                dropped.add(id(b))
    keep = [v for v in keep if id(v) not in dropped]
    # 2. items inside furniture
    furniture = [v for v in keep if v.entry.level == "furniture"]
    keep = [
        v
        for v in keep
        if not (v.entry.level == "item" and any(inside_fraction(v.seg, f.seg) > ITEM_IN_FURNITURE_FRACTION for f in furniture))
    ]
    # 3. outside the walls
    if len(room.walls) >= 2:
        keep = [v for v in keep if bool(room.inside_xz(v.seg.centre()[None], OUTSIDE_MARGIN_M)[0])]
    # 4. tall wall-flush slabs that are not door / window
    keep = [
        v
        for v in keep
        if v.entry.en in ("door", "window") or not (_wall_fraction(v.seg, room) >= WALL_SLAB_FRACTION and v.seg.size_lwh[2] > WALL_SLAB_MIN_H)
    ]
    return keep
