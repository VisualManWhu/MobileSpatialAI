import io
import time

import pytest
import torch

from recon3d.capture.format import CaptureFormatError
from recon3d.jobs.store import JobStore, UploadTooLargeError
from recon3d.jobs.worker import Worker


def wait_until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "data")


@pytest.fixture
def make_worker(store):
    workers = []

    def make(pipeline):
        w = Worker(store, pipeline)
        workers.append(w)
        return w

    yield make
    for w in workers:
        w.stop()


def test_create_writes_capture_and_queued_status(store):
    j = store.create(io.BytesIO(b"hello"))
    assert len(j) == 12 and all(c in "0123456789abcdef" for c in j)
    d = store.job_dir(j)
    assert (d / "capture.zip").read_bytes() == b"hello"
    s = store.get(j)
    assert (s.job_id, s.state, s.progress) == (j, "queued", 0.0)
    assert s.message
    assert "queued" in (d / "log.txt").read_text(encoding="utf-8")


def test_get_unknown_or_malicious_id_is_none(store):
    assert store.get("nope") is None
    assert store.get("../../etc") is None
    assert store.get("0123456789ab") is None


def test_update_is_atomic_no_temp_left(store):
    j = store.create(io.BytesIO(b"a"))
    store.update(j, state="training", progress=0.5, message="训练中")
    s = store.get(j)
    assert (s.state, s.progress, s.message) == ("training", 0.5, "训练中")
    assert sorted(p.name for p in store.job_dir(j).iterdir()) == ["capture.zip", "log.txt", "status.json"]


def test_create_rejects_oversize_and_cleans_up(tmp_path):
    store = JobStore(tmp_path / "data", max_upload_bytes=10)
    with pytest.raises(UploadTooLargeError):
        store.create(io.BytesIO(b"x" * 11))
    assert list((tmp_path / "data" / "jobs").iterdir()) == []
    assert store.create(io.BytesIO(b"x" * 10))


def test_jobs_run_in_order(store, make_worker):
    order = []
    w = make_worker(lambda d, p: order.append(d.name))
    w.start()
    a, b = store.create(io.BytesIO(b"a")), store.create(io.BytesIO(b"b"))
    wait_until(lambda: store.get(b).state == "done")
    assert order == [a, b]
    assert store.get(a).progress == 1.0


def test_oom_marks_failed_and_continues(store, make_worker):
    calls = iter([torch.cuda.OutOfMemoryError(), None])

    def pipe(d, p):
        e = next(calls)
        if e:
            raise e

    w = make_worker(pipe)
    w.start()
    a, b = store.create(io.BytesIO(b"a")), store.create(io.BytesIO(b"b"))
    wait_until(lambda: store.get(b).state == "done")
    assert store.get(a).state == "failed" and "显存不足" in store.get(a).message


def test_capture_format_error_uses_chinese_message(store, make_worker):
    def pipe(d, p):
        raise CaptureFormatError("拍摄包不是有效的 zip 文件")

    make_worker(pipe).start()
    j = store.create(io.BytesIO(b"a"))
    wait_until(lambda: store.get(j).state == "failed")
    assert store.get(j).message == "拍摄包不是有效的 zip 文件"


def test_other_error_message_and_traceback_logged(store, make_worker):
    def pipe(d, p):
        raise ValueError("boom")

    make_worker(pipe).start()
    j = store.create(io.BytesIO(b"a"))
    wait_until(lambda: store.get(j).state == "failed")
    assert store.get(j).message == "重建失败：ValueError"
    log = (store.job_dir(j) / "log.txt").read_text(encoding="utf-8")
    assert "Traceback" in log and "boom" in log


def test_progress_maps_to_states(store, make_worker):
    seen = []

    def pipe(d, p):
        p(0.45)
        seen.append(store.get(d.name))
        p(0.95)
        seen.append(store.get(d.name))

    make_worker(pipe).start()
    j = store.create(io.BytesIO(b"a"))
    wait_until(lambda: store.get(j).state == "done")
    assert (seen[0].state, seen[0].progress) == ("training", 0.45)
    assert (seen[1].state, seen[1].progress) == ("exporting", 0.95)


def test_restart_recovery(store):
    j = store.create(io.BytesIO(b"a"))
    store.update(j, state="training")
    q = store.create(io.BytesIO(b"b"))
    d = store.create(io.BytesIO(b"c"))
    store.update(d, state="done", progress=1.0)
    JobStore(store.data_dir).recover_after_restart()
    assert store.get(j).state == "failed" and "服务重启" in store.get(j).message
    assert store.get(q).state == "failed"
    assert store.get(d).state == "done"


def test_worker_stop_joins_thread(store):
    w = Worker(store, lambda d, p: None)
    w.start()
    w.stop()
    assert not w.is_alive()


def test_oom_detected_from_runtime_error_message(store, make_worker):
    def pipe(d, p):
        raise RuntimeError("CUDA error: out of memory")

    make_worker(pipe).start()
    j = store.create(io.BytesIO(b"a"))
    wait_until(lambda: store.get(j).state == "failed")
    assert "显存不足" in store.get(j).message


def test_oom_detection_paths():
    from recon3d.jobs.worker import is_oom

    class Sub(torch.cuda.OutOfMemoryError):
        pass

    assert is_oom(torch.cuda.OutOfMemoryError())
    assert is_oom(Sub())  # matched through the MRO, not the exact class name
    assert is_oom(RuntimeError("CUDA error: out of memory"))
    assert is_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED: Out Of Memory"))
    assert not is_oom(RuntimeError("shape mismatch"))


def test_start_twice_runs_one_thread(store):
    import threading

    w = Worker(store, lambda d, p: None)
    w.start()
    w.start()
    try:
        assert sum(t.name == "recon3d-worker" for t in threading.enumerate()) == 1
    finally:
        w.stop()
    assert not any(t.name == "recon3d-worker" for t in threading.enumerate())
