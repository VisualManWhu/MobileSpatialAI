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
