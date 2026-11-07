import hashlib
import io
import os
import json
import threading
import time

import pytest
import torch

from recon3d.capture.format import CaptureFormatError
from recon3d.jobs.store import (
    ChecksumMismatch,
    IncompleteUpload,
    JobActive,
    JobStore,
    OffsetMismatch,
    UploadBusy,
    UploadInProgress,
    UploadTooLargeError,
)
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
    w = make_worker(lambda d, m, p: order.append(d.name))
    w.start()
    a, b = store.create(io.BytesIO(b"a")), store.create(io.BytesIO(b"b"))
    wait_until(lambda: store.get(b).state == "done")
    assert order == [a, b]
    assert store.get(a).progress == 1.0


def test_oom_marks_failed_and_continues(store, make_worker):
    calls = iter([torch.cuda.OutOfMemoryError(), None])

    def pipe(d, m, p):
        e = next(calls)
        if e:
            raise e

    w = make_worker(pipe)
    w.start()
    a, b = store.create(io.BytesIO(b"a")), store.create(io.BytesIO(b"b"))
    wait_until(lambda: store.get(b).state == "done")
    assert store.get(a).state == "failed" and "显存不足" in store.get(a).message


def test_capture_format_error_uses_chinese_message(store, make_worker):
    def pipe(d, m, p):
        raise CaptureFormatError("拍摄包不是有效的 zip 文件")

    make_worker(pipe).start()
    j = store.create(io.BytesIO(b"a"))
    wait_until(lambda: store.get(j).state == "failed")
    assert store.get(j).message == "拍摄包不是有效的 zip 文件"


def test_other_error_message_and_traceback_logged(store, make_worker):
    def pipe(d, m, p):
        raise ValueError("boom")

    make_worker(pipe).start()
    j = store.create(io.BytesIO(b"a"))
    wait_until(lambda: store.get(j).state == "failed")
    msg = store.get(j).message
    assert msg == f"重建失败，请重新拍摄后上传；如仍失败，把服务端 data/jobs/{j}/log.txt 发给开发者"
    assert "boom" not in msg and "ValueError" not in msg
    log = (store.job_dir(j) / "log.txt").read_text(encoding="utf-8")
    assert "Traceback" in log and "boom" in log


def test_progress_maps_to_states(store, make_worker):
    seen = []

    def pipe(d, m, p):
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
    w = Worker(store, lambda d, m, p: None)
    w.start()
    w.stop()
    assert not w.is_alive()


def test_oom_detected_from_runtime_error_message(store, make_worker):
    def pipe(d, m, p):
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

    w = Worker(store, lambda d, m, p: None)
    w.start()
    w.start()
    try:
        assert sum(t.name == "recon3d-worker" for t in threading.enumerate()) == 1
    finally:
        w.stop()
    assert not any(t.name == "recon3d-worker" for t in threading.enumerate())


def test_list_newest_first_and_skips_junk_dirs(store):
    a = store.create(io.BytesIO(b"a"))
    time.sleep(0.01)
    b = store.create(io.BytesIO(b"b"))
    (store.jobs_dir / "deadbeef0000").mkdir()  # aborted upload leftover: no status.json
    (store.jobs_dir / ".DS_Store").write_bytes(b"")
    assert [s.job_id for s in store.list()] == [b, a]


# --- resumable uploads (docs/superpowers/specs/2026-11-05-recon3d-resumable-transfer-design.md) ---


def _rstore(tmp_path, **kw):
    return JobStore(tmp_path / "data", **kw)


def test_resumable_roundtrip(tmp_path):
    s = _rstore(tmp_path)
    data = b"abc" * 1000
    info = s.begin_resumable(len(data), hashlib.sha256(data).hexdigest())
    assert info.received == 0 and len(info.upload_id) == 12
    assert s.append_resumable(info.upload_id, 0, [data[:1000]]) == 1000
    assert s.resumable(info.upload_id).received == 1000
    assert s.append_resumable(info.upload_id, 1000, [data[1000:]]) == 3000
    jid = s.finish_resumable(info.upload_id)
    assert jid == info.upload_id and s.get(jid).state == "queued"
    assert (s.job_dir(jid) / "capture.zip").read_bytes() == data
    assert not (s.job_dir(jid) / "upload.json").exists() and s.resumable(jid) is None
    assert s.finish_resumable(jid) == jid  # idempotent
    assert s.queue.get_nowait() == jid and s.queue.empty()


def test_resumable_rejects_bad_begin(tmp_path):
    s = _rstore(tmp_path, max_upload_bytes=10)
    with pytest.raises(UploadTooLargeError):
        s.begin_resumable(11, "0" * 64)
    with pytest.raises(ValueError):
        s.begin_resumable(0, "0" * 64)
    with pytest.raises(ValueError):
        s.begin_resumable(5, "xyz")


def test_resumable_offset_mismatch_reports_received(tmp_path):
    s = _rstore(tmp_path)
    i = s.begin_resumable(10, "0" * 64)
    s.append_resumable(i.upload_id, 0, [b"12345"])
    with pytest.raises(OffsetMismatch) as e:
        s.append_resumable(i.upload_id, 3, [b"x"])
    assert e.value.received == 5


def test_resumable_partial_iterable_keeps_bytes(tmp_path):
    s = _rstore(tmp_path)
    i = s.begin_resumable(10, "0" * 64)

    def chunks():
        yield b"1234"
        raise ConnectionError

    with pytest.raises(ConnectionError):
        s.append_resumable(i.upload_id, 0, chunks())
    assert s.resumable(i.upload_id).received == 4


def test_resumable_over_declared_size_deletes(tmp_path):
    s = _rstore(tmp_path)
    i = s.begin_resumable(4, "0" * 64)
    with pytest.raises(UploadTooLargeError):
        s.append_resumable(i.upload_id, 0, [b"12345"])
    assert s.resumable(i.upload_id) is None and not s.job_dir(i.upload_id).exists()


def test_resumable_finish_incomplete_and_checksum(tmp_path):
    s = _rstore(tmp_path)
    i = s.begin_resumable(4, "0" * 64)
    s.append_resumable(i.upload_id, 0, [b"12"])
    with pytest.raises(IncompleteUpload) as e:
        s.finish_resumable(i.upload_id)
    assert e.value.received == 2
    s.append_resumable(i.upload_id, 2, [b"34"])
    with pytest.raises(ChecksumMismatch):
        s.finish_resumable(i.upload_id)
    assert not s.job_dir(i.upload_id).exists()


def test_resumable_busy(tmp_path):
    s = _rstore(tmp_path)
    i = s.begin_resumable(10, "0" * 64)
    started, release = threading.Event(), threading.Event()

    def slow():
        yield b"1"
        started.set()
        release.wait(5)
        yield b"2"

    t = threading.Thread(target=lambda: s.append_resumable(i.upload_id, 0, slow()))
    t.start()
    assert started.wait(5)
    with pytest.raises(UploadBusy) as e:
        s.append_resumable(i.upload_id, 1, [b"x"])
    assert e.value.received == 1
    release.set()
    t.join(5)
    assert s.resumable(i.upload_id).received == 2


def test_sweep_and_startup_correct_received_from_part_size(tmp_path):
    s = _rstore(tmp_path)
    old = s.begin_resumable(10, "0" * 64)
    fresh = s.begin_resumable(10, "0" * 64)
    meta = s.job_dir(old.upload_id) / "upload.json"
    j = json.loads(meta.read_text(encoding="utf-8"))
    j["updated_at"] -= 90000
    meta.write_text(json.dumps(j), encoding="utf-8")
    # A crash after bytes landed but before upload.json was updated.
    (s.job_dir(fresh.upload_id) / "capture.zip.part").write_bytes(b"123")
    assert s.sweep_resumable() == 1
    assert s.resumable(old.upload_id) is None and not s.job_dir(old.upload_id).exists()
    s.recover_after_restart()
    assert s.resumable(fresh.upload_id).received == 3


def test_sweep_skips_upload_being_received(tmp_path):
    s = _rstore(tmp_path)
    i = s.begin_resumable(10, "0" * 64)
    meta = s.job_dir(i.upload_id) / "upload.json"
    j = json.loads(meta.read_text(encoding="utf-8"))
    j["updated_at"] -= 90000
    meta.write_text(json.dumps(j), encoding="utf-8")
    started, release = threading.Event(), threading.Event()

    def slow():
        yield b"1"
        started.set()
        release.wait(5)

    t = threading.Thread(target=lambda: s.append_resumable(i.upload_id, 0, slow()))
    t.start()
    assert started.wait(5)
    assert s.sweep_resumable() == 0
    release.set()
    t.join(5)
    assert s.resumable(i.upload_id).received == 1


def test_finish_waiting_behind_another_finish_is_idempotent(tmp_path):
    """Client retries finish while the first one is still hashing: the second must not see the renamed
    part as "0 bytes received" (review finding I3)."""
    s = _rstore(tmp_path)
    data = b"abcd"
    i = s.begin_resumable(4, hashlib.sha256(data).hexdigest())
    s.append_resumable(i.upload_id, 0, [data])
    lock = s._upload_lock(i.upload_id)
    result = {}
    lock.acquire()
    t = threading.Thread(target=lambda: result.update(second=s.finish_resumable(i.upload_id)))
    t.start()
    time.sleep(0.2)  # second finish is now blocked on the lock, having read the metadata
    # First finish (simulated inline while holding the lock).
    d = s.job_dir(i.upload_id)
    os.replace(d / "capture.zip.part", d / "capture.zip")
    (d / "upload.json").unlink()
    s._commit(i.upload_id, 4)
    lock.release()
    t.join(5)
    assert result["second"] == i.upload_id


def test_list_resumable_newest_first_excludes_finished(tmp_path):
    s = _rstore(tmp_path)
    data = b"q" * 300
    a = s.begin_resumable(300, hashlib.sha256(data).hexdigest())
    time.sleep(0.01)
    b = s.begin_resumable(300, hashlib.sha256(data).hexdigest())
    s.append_resumable(a.upload_id, 0, [data[:100]])
    done = s.begin_resumable(300, hashlib.sha256(data).hexdigest())
    s.append_resumable(done.upload_id, 0, [data])
    s.finish_resumable(done.upload_id)
    got = s.list_resumable()
    assert [i.upload_id for i in got] == [b.upload_id, a.upload_id]
    assert got[1].received == 100


# --- delete (docs/superpowers/specs/2026-11-05-recon3d-delete-design.md) ---


def test_delete_removes_finished_job_dir(store):
    j = store.create(io.BytesIO(b"x"))
    store.update(j, state="done", progress=1.0, message="完成")
    store.delete(j)
    assert not store.job_dir(j).exists() and store.get(j) is None
    with pytest.raises(KeyError):
        store.delete(j)


def test_delete_refuses_training_job(store):
    j = store.create(io.BytesIO(b"x"))
    store.update(j, state="training", progress=0.3, message="训练中")
    with pytest.raises(JobActive):
        store.delete(j)
    assert store.get(j).state == "training"


def test_delete_resumable_upload_and_busy(tmp_path):
    s = _rstore(tmp_path)
    info = s.begin_resumable(10, "0" * 64)
    lock = s._upload_lock(info.upload_id)
    lock.acquire()
    try:
        with pytest.raises(UploadBusy):
            s.delete(info.upload_id)
    finally:
        lock.release()
    s.delete(info.upload_id)
    assert s.resumable(info.upload_id) is None and not s.job_dir(info.upload_id).exists()


def test_delete_rejects_bad_ids(store):
    for bad in ("nope", "../../etc", "ABCDEF123456"):
        with pytest.raises(KeyError):
            store.delete(bad)


def test_delete_queued_job_is_skipped_by_worker(store, make_worker):
    ran = []
    j1 = store.create(io.BytesIO(b"a"))
    j2 = store.create(io.BytesIO(b"b"))
    store.delete(j1)
    make_worker(lambda d, m, p: ran.append(d.name)).start()
    wait_until(lambda: store.get(j2) is not None and store.get(j2).state == "done")
    assert ran == [j2] and not store.job_dir(j1).exists()


def test_status_mode_defaults_to_fast_for_old_status_json(store):
    j = store.create(io.BytesIO(b"x"))
    d = store.job_dir(j)
    raw = json.loads((d / "status.json").read_text(encoding="utf-8"))
    raw.pop("mode")
    (d / "status.json").write_text(json.dumps(raw), encoding="utf-8")
    assert store.get(j).mode == "fast"


def test_worker_passes_mode_and_stage_messages(store, make_worker):
    j = store.create(io.BytesIO(b"x"))
    store.update(j, mode="precise")
    seen = []

    def pipe(d, mode, on_progress):
        seen.append(mode)
        on_progress(0.1, "位姿优化 (特征提取)")
        seen.append(store.get(j).message)
        on_progress(0.5, None)
        seen.append(store.get(j).message)
        return "完成（高精位姿失败，已按快速模式重建）"

    w = make_worker(pipe)
    w.start()
    wait_until(lambda: store.get(j).state == "done")
    assert seen == ["precise", "位姿优化 (特征提取)", "训练中"]
    assert store.get(j).message == "完成（高精位姿失败，已按快速模式重建）"
    assert store.get(j).progress == 1.0


def test_worker_default_done_message_when_pipeline_returns_none(store, make_worker):
    j = store.create(io.BytesIO(b"x"))
    w = make_worker(lambda d, m, p: None)
    w.start()
    wait_until(lambda: store.get(j).state == "done")
    assert store.get(j).message == "完成"


def test_create_and_resumable_store_mode(tmp_path):
    s = JobStore(tmp_path / "d")
    j = s.create(io.BytesIO(b"x"), mode="precise")
    assert s.get(j).mode == "precise"
    data = b"abc"
    info = s.begin_resumable(len(data), hashlib.sha256(data).hexdigest(), mode="precise")
    assert s.resumable(info.upload_id).mode == "precise"
    s.append_resumable(info.upload_id, 0, [data])
    assert s.get(s.finish_resumable(info.upload_id)).mode == "precise"
    with pytest.raises(ValueError):
        s.create(io.BytesIO(b"x"), mode="turbo")


def test_resumable_mode_defaults_to_fast_for_old_upload_json(tmp_path):
    s = JobStore(tmp_path / "d")
    info = s.begin_resumable(3, "0" * 64)
    meta = s.job_dir(info.upload_id) / "upload.json"
    raw = json.loads(meta.read_text(encoding="utf-8"))
    raw.pop("mode")
    meta.write_text(json.dumps(raw), encoding="utf-8")
    assert s.resumable(info.upload_id).mode == "fast"


def test_rerun_hardlinks_capture_and_queues(store):
    src = store.create(io.BytesIO(b"zipbytes"))
    new = store.rerun(src, "precise")
    assert new != src
    d = store.job_dir(new)
    assert (d / "capture.zip").read_bytes() == b"zipbytes"
    assert os.stat(d / "capture.zip").st_ino == os.stat(store.job_dir(src) / "capture.zip").st_ino  # hard link
    s = store.get(new)
    assert (s.state, s.mode, s.progress) == ("queued", "precise", 0.0)
    first = (d / "log.txt").read_text(encoding="utf-8").splitlines()[0]
    assert first.endswith(f"created from {src} (rerun, mode=precise)")
    assert store.queue.get_nowait() == src and store.queue.get_nowait() == new


def test_rerun_missing_or_uploading_source(tmp_path):
    s = JobStore(tmp_path / "d")
    with pytest.raises(KeyError):
        s.rerun("000000000000", "fast")
    info = s.begin_resumable(3, "0" * 64)
    with pytest.raises(UploadInProgress):
        s.rerun(info.upload_id, "fast")
    with pytest.raises(ValueError):
        s.rerun(s.create(io.BytesIO(b"x")), "turbo")


def test_rerun_copies_when_link_fails(store, monkeypatch):
    src = store.create(io.BytesIO(b"zipbytes"))

    def no_link(a, b):
        raise OSError("EXDEV")

    monkeypatch.setattr(os, "link", no_link)
    new = store.rerun(src, "fast")
    assert (store.job_dir(new) / "capture.zip").read_bytes() == b"zipbytes"
