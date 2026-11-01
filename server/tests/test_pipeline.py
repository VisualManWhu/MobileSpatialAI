import shutil
from dataclasses import replace

import pytest

from recon3d.capture.format import CaptureFormatError
from recon3d.config import PROFILES
from recon3d.jobs.pipeline import run_pipeline
from tests.fixtures.make_capture import make_capture


def test_bad_zip_raises_capture_format_error(tmp_path):
    (tmp_path / "capture.zip").write_bytes(b"not a zip")
    with pytest.raises(CaptureFormatError):
        run_pipeline(tmp_path, PROFILES["laptop"], lambda p: None)
    assert not (tmp_path / "capture").exists()  # extraction dir cleaned up


@pytest.mark.gpu
def test_pipeline_end_to_end(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    shutil.copy(make_capture(tmp_path, n_frames=4), job / "capture.zip")
    profile = replace(PROFILES["laptop"], iterations=200)
    seen = []
    run_pipeline(job, profile, seen.append)
    assert (job / "scene.splat").stat().st_size > 0
    assert (job / "scene.ply").stat().st_size > 0
    assert not (job / "capture").exists()
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert any(abs(p - 0.9) < 1e-9 for p in seen)  # training ends at 0.9, export -> 1.0
