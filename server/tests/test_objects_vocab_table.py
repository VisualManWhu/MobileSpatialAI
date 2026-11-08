import json
import zipfile

import numpy as np
import pytest

from recon3d.objects import table, vocab


V1 = {"cup", "bottle", "book", "remote control", "chair", "table", "monitor", "laptop", "keyboard",
      "mouse", "phone", "box", "bag", "lamp", "plant", "whiteboard", "door", "window", "trash can",
      "pen", "sofa", "bed", "pillow", "shelf", "cabinet", "tv", "speaker", "headphones", "clock",
      "picture frame", "curtain", "mirror", "fan", "printer", "backpack", "shoes", "towel", "sink",
      "refrigerator", "microwave"}


def test_v1_vocab_is_preserved():
    assert V1 <= {e.en for e in vocab.VOCAB}
    assert 55 <= len(vocab.VOCAB) <= 70
    assert vocab.zh_label("cup") == "杯子" and vocab.zh_label("remote control") == "遥控器"
    assert all(e.zh and not e.zh.isascii() for e in vocab.VOCAB)
    assert vocab.zh_label("door") == "门" and vocab.entry("door").level == "structure"
    assert {e.level for e in vocab.VOCAB} == {"structure", "furniture", "item"}
    assert len({e.en for e in vocab.VOCAB}) == len(vocab.VOCAB)


def test_candidates_use_size_and_support():
    # a 2.0 m tall, 0.9 m wide thing flush with a wall: door and window pass the size/support gate
    # (the floor-distance split between them is verify.door_window_ok), phone does not
    c = {e.en for e in vocab.candidates((0.9, 0.1, 2.0), "wall:1", "structure")}
    assert {"door", "window"} <= c
    assert "phone" not in {e.en for e in vocab.candidates((0.9, 0.1, 2.0), "floor", "item")}
    # a 15 cm flat thing on a table: phone yes, remote yes, monitor no
    c = {e.en for e in vocab.candidates((0.15, 0.07, 0.01), "on:3", "item")}
    assert {"phone", "remote control"} <= c and "monitor" not in c
    # voxel-measured sizes run a cell large: a 21 cm "phone" still qualifies (3 cm tolerance)
    assert "phone" in {e.en for e in vocab.candidates((0.21, 0.1, 0.04), "on:3", "item")}
    # "none" support matches everything that fits the size
    assert "chair" in {e.en for e in vocab.candidates((0.5, 0.5, 0.9), "none", "furniture")}
    # by_level partitions the vocab
    assert sum(len(vocab.by_level(lv)) for lv in ("structure", "furniture", "item")) == len(vocab.VOCAB)


def test_every_entry_has_sane_ranges():
    for e in vocab.VOCAB:
        assert 0 < e.h_min < e.h_max and 0 < e.l_min < e.l_max and e.supports, e.en
        assert e.supports <= {"floor", "wall", "ceiling", "on"}, e.en


def test_prompt_batches_cover_vocab_in_groups_of_20():
    batches = vocab.prompt_batches()
    n = len(vocab.VOCAB)
    assert len(batches) == -(-n // 20) and all(b.endswith(".") for b in batches)
    assert sum(b.count(". ") + 1 for b in batches) == n


def test_label_of_maps_detector_phrases_back_to_vocab():
    assert vocab.label_of("remote control phone") == "remote control"
    assert vocab.label_of("trash") is None and vocab.label_of("trash can") == "trash can"
    assert vocab.label_of("cup") == "cup"


def test_ids_roundtrip_little_endian_uint16():
    ids = np.array([0, 1, 65535, 7], dtype=np.uint16)
    data = table.encode_ids(ids)
    assert data == b"\x00\x00\x01\x00\xff\xff\x07\x00"
    assert np.array_equal(table.decode_ids(data), ids)
    with pytest.raises(ValueError):
        table.decode_ids(b"\x01")


def test_write_objects_produces_json_bin_and_zip(tmp_path):
    out = tmp_path / "job" / "objects"
    (out / "thumbs").mkdir(parents=True)
    (out / "thumbs" / "1.jpg").write_bytes(b"jpg")
    rec = table.ObjectRecord(
        1, "chair", "椅子", 0.64, 17, (1.0, 0.5, -2.0), (0.7, 0.0, -2.3), (1.3, 1.0, -1.7), 3,
        level="furniture", size=(0.6, 0.6, 1.0), support="floor", confidence=0.75,
    )
    ids = np.array([1, 0, 1, 1], dtype=np.uint16)
    structure = {"floor_y": 0.0, "ceiling_y": None, "walls": [{"id": 1, "normal": [1, 0, 0], "d": 2.0, "u": [0, 0, 1], "extent": [-2, 2, 0, 2.6]}]}
    zpath = table.write_objects(
        out, "abc123", frames_used=12, ids=ids, objects=[rec], relations=[{"a": 1, "rel": "on", "b": 2}], structure=structure
    )
    assert zpath == tmp_path / "job" / "objects.zip"
    j = json.loads((out / "objects.json").read_text(encoding="utf-8"))
    assert j["version"] == 2 and j["job_id"] == "abc123" and j["frames_used"] == 12 and j["gaussians"] == 4
    assert j["structure"] == structure
    o = j["objects"][0]
    assert o["id"] == 1 and o["label_zh"] == "椅子" and o["thumb"] == "thumbs/1.jpg" and o["gaussian_count"] == 3
    assert o["bbox"] == {"min": [0.7, 0.0, -2.3], "max": [1.3, 1.0, -1.7]}
    assert o["level"] == "furniture" and o["size"] == [0.6, 0.6, 1.0] and o["support"] == "floor" and o["confidence"] == 0.75
    assert j["relations"] == [{"a": 1, "rel": "on", "b": 2}]
    assert (out / "objects.bin").read_bytes() == table.encode_ids(ids)
    assert sorted(zipfile.ZipFile(zpath).namelist()) == ["objects.bin", "objects.json", "thumbs/1.jpg"]
