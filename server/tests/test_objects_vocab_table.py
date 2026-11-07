import json
import zipfile

import numpy as np
import pytest

from recon3d.objects import table, vocab


def test_vocab_has_40_entries_with_chinese_labels():
    assert len(vocab.VOCAB) == 40
    assert vocab.zh_label("cup") == "杯子" and vocab.zh_label("remote control") == "遥控器"
    assert all(zh and not zh.isascii() for _, zh in vocab.VOCAB)


def test_prompt_batches_cover_vocab_in_groups_of_20():
    batches = vocab.prompt_batches()
    assert len(batches) == 2 and all(b.endswith(".") for b in batches)
    assert sum(b.count(". ") + 1 for b in batches) == 40


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
    rec = table.ObjectRecord(1, "chair", "椅子", 0.64, 17, (1.0, 0.5, -2.0), (0.7, 0.0, -2.3), (1.3, 1.0, -1.7), 3)
    ids = np.array([1, 0, 1, 1], dtype=np.uint16)
    zpath = table.write_objects(
        out, "abc123", frames_used=12, ids=ids, objects=[rec], relations=[{"a": 1, "rel": "on", "b": 2}]
    )
    assert zpath == tmp_path / "job" / "objects.zip"
    j = json.loads((out / "objects.json").read_text(encoding="utf-8"))
    assert j["version"] == 1 and j["job_id"] == "abc123" and j["frames_used"] == 12 and j["gaussians"] == 4
    o = j["objects"][0]
    assert o["id"] == 1 and o["label_zh"] == "椅子" and o["thumb"] == "thumbs/1.jpg" and o["gaussian_count"] == 3
    assert o["bbox"] == {"min": [0.7, 0.0, -2.3], "max": [1.3, 1.0, -1.7]}
    assert j["relations"] == [{"a": 1, "rel": "on", "b": 2}]
    assert (out / "objects.bin").read_bytes() == table.encode_ids(ids)
    assert sorted(zipfile.ZipFile(zpath).namelist()) == ["objects.bin", "objects.json", "thumbs/1.jpg"]
