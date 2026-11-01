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
        load_capture(make_capture(tmp_path, format_version=2), tmp_path / "w")


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
