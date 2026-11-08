"""Fixed indoor vocabulary (v2): every class carries its level, a plausible size range and the
supports it may rest on. The ranges are the first gate a 3D segment must pass before the VLM is
even asked (spec 2026-11-08 §4.3); the English name is also the Grounding DINO prompt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Level = Literal["structure", "furniture", "item"]

FLOOR, WALL, CEILING, ON = "floor", "wall", "ceiling", "on"


@dataclass(frozen=True)
class VocabEntry:
    en: str
    zh: str
    level: Level
    h_min: float  # height range, metres
    h_max: float
    l_min: float  # longest horizontal side range, metres
    l_max: float
    supports: frozenset[str]  # subset of {floor, wall, ceiling, on}


def _e(en, zh, level, h, l, *supports) -> VocabEntry:
    return VocabEntry(en, zh, level, h[0], h[1], l[0], l[1], frozenset(supports))


# fmt: off
VOCAB: list[VocabEntry] = [
    # --- structure (floor / ceiling / walls come from geometry, not from this list) ---
    _e("door", "门", "structure", (1.8, 2.4), (0.6, 1.2), FLOOR, WALL),
    _e("window", "窗", "structure", (0.4, 2.2), (0.3, 3.0), WALL),
    # --- furniture ---
    _e("table", "桌子", "furniture", (0.4, 1.1), (0.6, 3.0), FLOOR),
    _e("desk", "书桌", "furniture", (0.6, 0.9), (0.8, 2.0), FLOOR),
    _e("coffee table", "茶几", "furniture", (0.3, 0.6), (0.5, 1.5), FLOOR),
    _e("chair", "椅子", "furniture", (0.7, 1.3), (0.4, 0.8), FLOOR),
    _e("armchair", "扶手椅", "furniture", (0.6, 1.1), (0.6, 1.1), FLOOR),
    _e("stool", "凳子", "furniture", (0.3, 0.8), (0.25, 0.6), FLOOR),
    _e("bench", "长凳", "furniture", (0.3, 0.6), (0.8, 2.5), FLOOR),
    _e("sofa", "沙发", "furniture", (0.6, 1.1), (1.2, 3.5), FLOOR),
    _e("bed", "床", "furniture", (0.3, 1.2), (1.6, 2.4), FLOOR),
    _e("nightstand", "床头柜", "furniture", (0.4, 0.8), (0.3, 0.7), FLOOR),
    _e("cabinet", "柜子", "furniture", (0.5, 2.4), (0.3, 2.5), FLOOR, WALL),
    _e("wardrobe", "衣柜", "furniture", (1.6, 2.6), (0.6, 3.0), FLOOR),
    _e("shelf", "架子", "furniture", (0.3, 2.4), (0.3, 2.0), FLOOR, WALL),
    _e("bookshelf", "书架", "furniture", (0.8, 2.4), (0.5, 2.0), FLOOR, WALL),
    _e("refrigerator", "冰箱", "furniture", (0.8, 2.1), (0.5, 1.0), FLOOR),
    _e("whiteboard", "白板", "furniture", (0.6, 1.5), (0.8, 3.0), WALL, FLOOR),
    _e("tv", "电视", "furniture", (0.3, 1.2), (0.5, 2.0), WALL, ON, FLOOR),
    _e("curtain", "窗帘", "furniture", (1.0, 3.0), (0.5, 4.0), WALL, CEILING),
    _e("mirror", "镜子", "furniture", (0.3, 2.0), (0.2, 1.5), WALL, FLOOR),
    _e("printer", "打印机", "furniture", (0.15, 0.6), (0.3, 0.7), ON, FLOOR),
    _e("trash can", "垃圾桶", "furniture", (0.2, 0.9), (0.2, 0.5), FLOOR),
    _e("floor lamp", "落地灯", "furniture", (1.2, 2.0), (0.2, 0.6), FLOOR),
    _e("radiator", "暖气片", "furniture", (0.3, 1.0), (0.4, 2.0), WALL, FLOOR),
    _e("air conditioner", "空调", "furniture", (0.2, 1.9), (0.5, 1.2), WALL, FLOOR),
    _e("water dispenser", "饮水机", "furniture", (0.8, 1.3), (0.3, 0.5), FLOOR),
    _e("sink", "水槽", "furniture", (0.1, 0.4), (0.4, 1.2), ON, FLOOR, WALL),
    # --- items ---
    _e("cup", "杯子", "item", (0.05, 0.2), (0.05, 0.15), ON, FLOOR),
    _e("bottle", "瓶子", "item", (0.1, 0.4), (0.04, 0.15), ON, FLOOR),
    _e("book", "书", "item", (0.01, 0.35), (0.1, 0.35), ON, FLOOR),
    _e("remote control", "遥控器", "item", (0.005, 0.25), (0.1, 0.25), ON, FLOOR),
    _e("monitor", "显示器", "item", (0.25, 0.7), (0.4, 1.0), ON, WALL, FLOOR),
    _e("laptop", "笔记本电脑", "item", (0.01, 0.35), (0.25, 0.45), ON, FLOOR),
    _e("keyboard", "键盘", "item", (0.01, 0.06), (0.25, 0.5), ON, FLOOR),
    _e("mouse", "鼠标", "item", (0.02, 0.06), (0.08, 0.14), ON, FLOOR),
    _e("phone", "手机", "item", (0.005, 0.2), (0.12, 0.2), ON, FLOOR),
    _e("tablet", "平板电脑", "item", (0.005, 0.3), (0.18, 0.35), ON, FLOOR),
    _e("box", "箱子", "item", (0.1, 0.8), (0.15, 1.0), ON, FLOOR),
    _e("bag", "包", "item", (0.1, 0.6), (0.15, 0.6), ON, FLOOR),
    _e("backpack", "背包", "item", (0.25, 0.6), (0.2, 0.45), ON, FLOOR),
    _e("lamp", "灯", "item", (0.15, 0.8), (0.1, 0.5), ON, FLOOR, CEILING, WALL),
    _e("ceiling light", "吸顶灯", "item", (0.05, 0.6), (0.2, 1.2), CEILING),
    _e("plant", "植物", "item", (0.1, 2.0), (0.1, 1.2), ON, FLOOR),
    _e("vase", "花瓶", "item", (0.1, 0.6), (0.05, 0.3), ON, FLOOR),
    _e("pen", "笔", "item", (0.005, 0.03), (0.1, 0.2), ON),
    _e("pillow", "枕头", "item", (0.05, 0.3), (0.3, 0.8), ON, FLOOR),
    _e("speaker", "音箱", "item", (0.05, 1.2), (0.05, 0.5), ON, FLOOR),
    _e("headphones", "耳机", "item", (0.03, 0.25), (0.1, 0.25), ON, FLOOR),
    _e("clock", "钟", "item", (0.05, 0.6), (0.05, 0.6), WALL, ON),
    _e("picture frame", "相框", "item", (0.1, 1.5), (0.1, 2.0), WALL, ON),
    _e("fan", "风扇", "item", (0.2, 1.5), (0.2, 0.6), ON, FLOOR, CEILING),
    _e("shoes", "鞋", "item", (0.05, 0.2), (0.2, 0.35), FLOOR, ON),
    _e("towel", "毛巾", "item", (0.05, 1.0), (0.2, 0.8), WALL, ON),
    _e("microwave", "微波炉", "item", (0.2, 0.4), (0.4, 0.6), ON, FLOOR),
    _e("kettle", "水壶", "item", (0.15, 0.35), (0.12, 0.3), ON),
    _e("bowl", "碗", "item", (0.03, 0.15), (0.08, 0.3), ON),
    _e("basket", "篮子", "item", (0.1, 0.5), (0.2, 0.6), ON, FLOOR),
    _e("tissue box", "纸巾盒", "item", (0.05, 0.15), (0.1, 0.3), ON),
    _e("router", "路由器", "item", (0.02, 0.3), (0.1, 0.35), ON, WALL),
]
# fmt: on

_BY_EN = {e.en: e for e in VOCAB}


def entry(en: str) -> VocabEntry:
    return _BY_EN[en]


def zh_label(en: str) -> str:
    return _BY_EN[en].zh


def by_level(level: Level) -> list[VocabEntry]:
    return [e for e in VOCAB if e.level == level]


def support_kind(support: str) -> str:
    """`on:<id>` -> `on`, `wall:<id>` -> `wall`, otherwise the string itself (floor / ceiling / none)."""
    return support.split(":", 1)[0]


def candidates(size_lwh: tuple[float, float, float], support: str, level: Level) -> list[VocabEntry]:
    """Entries of `level` whose size range contains the segment's height and longest side and whose
    supports allow where it rests; `none` support matches every entry."""
    l, w, h = size_lwh
    longest = max(l, w)
    kind = support_kind(support)
    return [
        e
        for e in by_level(level)
        if e.h_min <= h <= e.h_max and e.l_min <= longest <= e.l_max and (kind == "none" or kind in e.supports)
    ]


def prompt_batches(max_per_batch: int = 20) -> list[str]:
    """Grounding DINO text prompts: "cup. bottle. ... ." in batches (its token budget is small)."""
    names = [e.en for e in VOCAB]
    return [". ".join(names[i : i + max_per_batch]) + "." for i in range(0, len(names), max_per_batch)]


def label_of(phrase: str) -> str | None:
    """Vocab entry for a detector phrase such as "remote control phone": the longest entry that
    appears as whole words in the phrase; None when nothing matches ("trash" is not "trash can")."""
    words = phrase.lower().split()
    best: str | None = None
    for e in VOCAB:
        parts = e.en.split()
        n = len(parts)
        if any(words[i : i + n] == parts for i in range(len(words) - n + 1)):
            if best is None or len(e.en) > len(best):
                best = e.en
    return best
