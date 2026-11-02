import numpy as np
import pytest

from recon3d.capture.format import CaptureFormatError, load_capture
from tests.fixtures.make_capture import make_capture


def test_load_valid_capture(tmp_path):
    cap = load_capture(make_capture(tmp_path, n_frames=4), tmp_path / "w")
    assert cap.manifest.format_version == 1
    assert len(cap.frames) == 4
    assert cap.frames[0].pose.shape == (4, 4)


def test_rejects_unknown_version(tmp_path):
    with pytest.raises(CaptureFormatError, match="不支持的拍摄包版本"):
        load_capture(make_capture(tmp_path, format_version=3), tmp_path / "w")


def test_rejects_missing_image(tmp_path):
    with pytest.raises(CaptureFormatError, match="缺少图片"):
        load_capture(make_capture(tmp_path, drop_image=1), tmp_path / "w")


def test_frame_count_mismatch(tmp_path):
    with pytest.raises(CaptureFormatError, match="帧数不一致"):
        load_capture(make_capture(tmp_path, frame_count=9), tmp_path / "w")


def test_pose_translation_in_last_column(tmp_path):
    cap = load_capture(make_capture(tmp_path, n_frames=4), tmp_path / "w")
    pose = cap.frames[1].pose
    assert pose[0, 3] == pytest.approx(0.2)
    assert list(pose[3, :]) == [0, 0, 0, 1]


def _rewrite_frames(zip_path, fn):
    import json
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        items = {n: zf.read(n) for n in zf.namelist()}
    lines = items["frames.jsonl"].decode().splitlines()
    items["frames.jsonl"] = "\n".join(fn(lines)).encode()
    with zipfile.ZipFile(zip_path, "w") as zf:
        for n, b in items.items():
            zf.writestr(n, b)
    return zip_path


@pytest.mark.parametrize("bad", ["../../x.jpg", "/etc/passwd"])
def test_rejects_escaping_frame_path(tmp_path, bad):
    import json

    def fn(lines):
        d = json.loads(lines[0])
        d["image"] = bad
        return [json.dumps(d)] + lines[1:]

    z = _rewrite_frames(make_capture(tmp_path), fn)
    with pytest.raises(CaptureFormatError, match="路径非法"):
        load_capture(z, tmp_path / "w")


@pytest.mark.parametrize("bad", ["[1]", "5"])
def test_rejects_non_object_frame_line(tmp_path, bad):
    z = _rewrite_frames(make_capture(tmp_path), lambda lines: [bad] + lines[1:])
    with pytest.raises(CaptureFormatError, match="frames.jsonl 格式错误"):
        load_capture(z, tmp_path / "w")


def test_loads_android_package(tmp_path):
    """Cross-language contract: package written by the Android CapturePackageWriter (JUnit generates it)."""
    from pathlib import Path

    cap = load_capture(Path(__file__).parent / "fixtures" / "android_v1.zip", tmp_path / "w")
    assert cap.manifest.format_version == 1
    assert len(cap.frames) == 3
    assert cap.frames[1].pose.shape == (4, 4)
    assert abs(cap.frames[1].pose[0, 3] - 0.1) < 1e-6  # translation survives column-major round trip
    assert all(f.depth is not None for f in cap.frames)
    assert cap.manifest.depth_width == 8
    # optional depth intrinsics (depth-image pixels) written by the Android writer
    assert all(f.depth_fx == 6.5 and f.depth_fy == 6.5 for f in cap.frames)
    assert all(f.depth_cx == 4.0 and f.depth_cy == 3.0 for f in cap.frames)


def test_v2_with_confidence_loads(tmp_path):
    conf = [np.full((6, 8), 200, np.uint8)] * 2
    z = make_capture(tmp_path, n_frames=2, format_version=2, confidences=conf, depth_kind="raw",
                     depths=[np.full((6, 8), 1500, np.uint16)] * 2)
    cap = load_capture(z, tmp_path / "w")
    assert cap.manifest.format_version == 2
    assert cap.frames[0].confidence == "conf/000001.png"
    assert cap.frames[0].depth_kind == "raw"
    assert (cap.root / "conf/000001.png").is_file()


def test_v2_without_confidence_defaults_smooth(tmp_path):
    z = make_capture(tmp_path, n_frames=1, format_version=2)
    fr = load_capture(z, tmp_path / "w").frames[0]
    assert fr.confidence is None and fr.depth_kind == "smooth"


def test_v2_missing_confidence_file_is_error(tmp_path):
    z = make_capture(tmp_path, n_frames=1, format_version=2, confidences=[np.zeros((6, 8), np.uint8)],
                     depths=[np.zeros((6, 8), np.uint16)], drop_confidence=1)
    with pytest.raises(CaptureFormatError, match="缺少置信度图"):
        load_capture(z, tmp_path / "w")


def test_bad_depth_kind_is_error(tmp_path):
    z = make_capture(tmp_path, n_frames=1, format_version=2, depth_kind="fuzzy")
    with pytest.raises(CaptureFormatError, match="frames.jsonl 格式错误"):
        load_capture(z, tmp_path / "w")


def test_v3_rejected(tmp_path):
    with pytest.raises(CaptureFormatError, match="不支持的拍摄包版本: 3"):
        load_capture(make_capture(tmp_path, format_version=3), tmp_path / "w")
