"""Fixed open-vocabulary prompt list for indoor objects: (English prompt, Chinese display name)."""
from __future__ import annotations

VOCAB: list[tuple[str, str]] = [
    ("cup", "杯子"),
    ("bottle", "瓶子"),
    ("book", "书"),
    ("remote control", "遥控器"),
    ("chair", "椅子"),
    ("table", "桌子"),
    ("monitor", "显示器"),
    ("laptop", "笔记本电脑"),
    ("keyboard", "键盘"),
    ("mouse", "鼠标"),
    ("phone", "手机"),
    ("box", "箱子"),
    ("bag", "包"),
    ("lamp", "灯"),
    ("plant", "植物"),
    ("whiteboard", "白板"),
    ("door", "门"),
    ("window", "窗"),
    ("trash can", "垃圾桶"),
    ("pen", "笔"),
    ("sofa", "沙发"),
    ("bed", "床"),
    ("pillow", "枕头"),
    ("shelf", "架子"),
    ("cabinet", "柜子"),
    ("tv", "电视"),
    ("speaker", "音箱"),
    ("headphones", "耳机"),
    ("clock", "钟"),
    ("picture frame", "相框"),
    ("curtain", "窗帘"),
    ("mirror", "镜子"),
    ("fan", "风扇"),
    ("printer", "打印机"),
    ("backpack", "背包"),
    ("shoes", "鞋"),
    ("towel", "毛巾"),
    ("sink", "水槽"),
    ("refrigerator", "冰箱"),
    ("microwave", "微波炉"),
]

_ZH = dict(VOCAB)


def zh_label(en: str) -> str:
    return _ZH[en]


def prompt_batches(max_per_batch: int = 20) -> list[str]:
    """Grounding DINO text prompts: "cup. bottle. ... ." in batches (its token budget is small)."""
    names = [en for en, _ in VOCAB]
    return [". ".join(names[i : i + max_per_batch]) + "." for i in range(0, len(names), max_per_batch)]


def label_of(phrase: str) -> str | None:
    """Vocab entry for a detector phrase such as "remote control phone": the longest entry that
    appears as whole words in the phrase; None when nothing matches ("trash" is not "trash can")."""
    words = phrase.lower().split()
    best: str | None = None
    for en, _ in VOCAB:
        parts = en.split()
        n = len(parts)
        if any(words[i : i + n] == parts for i in range(len(words) - n + 1)):
            if best is None or len(en) > len(best):
                best = en
    return best
