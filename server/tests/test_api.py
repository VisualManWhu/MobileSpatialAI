import threading

import pytest
from fastapi.testclient import TestClient

from recon3d.api.app import create_app
from recon3d.config import PROFILES
from recon3d.jobs.store import JobStore
from recon3d.jobs.worker import Worker
from tests.test_jobs import wait_until


def _client(tmp_path, pipeline, **store_kw):
    store = JobStore(tmp_path / "data", **store_kw)
    app = create_app(store, Worker(store, pipeline), PROFILES["laptop"])
    return TestClient(app)


def _fast(d, p):
    (d / "scene.splat").write_bytes(b"\x01" * 64)


@pytest.fixture
def client(tmp_path):
    with _client(tmp_path, _fast) as c:
        yield c


@pytest.fixture
def client_slow(tmp_path):
    release = threading.Event()
    with _client(tmp_path, lambda d, p: release.wait()) as c:
        yield c
        release.set()


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "0.1.0" and body["profile"] == "laptop"
    assert isinstance(body["gpu"], str) and body["gpu"]


def test_api_upload_status_download(client):
    r = client.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")})
    assert r.status_code == 201
    jid = r.json()["job_id"]
    wait_until(lambda: client.get(f"/api/v1/jobs/{jid}").json()["state"] == "done")
    s = client.get(f"/api/v1/jobs/{jid}").json()
    assert s == {"job_id": jid, "state": "done", "progress": 1.0, "message": s["message"]}
    r = client.get(f"/api/v1/jobs/{jid}/scene.splat")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert len(r.content) == 64


def test_download_before_done_is_409(client_slow):
    jid = client_slow.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}).json()["job_id"]
    r = client_slow.get(f"/api/v1/jobs/{jid}/scene.splat")
    assert r.status_code == 409
    assert r.json() == {"detail": "任务尚未完成"}


def test_unknown_job_404(client):
    r = client.get("/api/v1/jobs/nope")
    assert r.status_code == 404
    assert r.json() == {"detail": "任务不存在"}
    assert client.get("/api/v1/jobs/nope/scene.splat").status_code == 404


def test_missing_capture_field_is_422(client):
    assert client.post("/api/v1/jobs", files={"other": ("c.zip", b"zip")}).status_code == 422


def test_upload_too_large_is_413(tmp_path):
    with _client(tmp_path, _fast, max_upload_bytes=100) as c:
        r = c.post("/api/v1/jobs", files={"capture": ("c.zip", b"x" * 500)})
        assert r.status_code == 413
        assert "上传" in r.json()["detail"]
        assert list((tmp_path / "data" / "jobs").iterdir()) == []


def test_startup_recovers_interrupted_jobs(tmp_path):
    import io

    store = JobStore(tmp_path / "data")
    j = store.create(io.BytesIO(b"a"))
    store.update(j, state="training")
    with _client(tmp_path, _fast) as c:
        s = c.get(f"/api/v1/jobs/{j}").json()
    assert s["state"] == "failed" and "服务重启" in s["message"]


BOUNDARY = "r3dboundary"


def _multipart(payload: bytes, name: str = "capture", closing: bool = True) -> bytes:
    head = (
        f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"c.zip\"\r\n"
        "Content-Type: application/zip\r\n\r\n"
    ).encode()
    return head + payload + (f"\r\n--{BOUNDARY}--\r\n".encode() if closing else b"")


def _chunked(body: bytes, size: int = 7):
    # A generator body makes httpx send Transfer-Encoding: chunked with no Content-Length.
    for i in range(0, len(body), size):
        yield body[i : i + size]


MP_HEADERS = {"content-type": f"multipart/form-data; boundary={BOUNDARY}"}


def test_declared_content_length_over_cap_is_413_before_body(tmp_path):
    with _client(tmp_path, _fast, max_upload_bytes=100) as c:
        headers = {**MP_HEADERS, "content-length": str(10 * 1024**2)}
        r = c.post("/api/v1/jobs", content=b"tiny", headers=headers)
        assert r.status_code == 413 and "上传" in r.json()["detail"]
        assert list((tmp_path / "data" / "jobs").iterdir()) == []


def test_chunked_oversize_is_413_and_nothing_left(tmp_path):
    with _client(tmp_path, _fast, max_upload_bytes=100) as c:
        body = _multipart(b"x" * 500)
        r = c.post("/api/v1/jobs", content=_chunked(body), headers=MP_HEADERS)
        assert r.status_code == 413
        assert list((tmp_path / "data" / "jobs").iterdir()) == []


def test_chunked_upload_streams_to_capture_zip(tmp_path):
    payload = bytes(range(256)) * 40  # crosses many tiny chunks and contains CR/LF bytes
    with _client(tmp_path, _fast) as c:
        r = c.post("/api/v1/jobs", content=_chunked(_multipart(payload)), headers=MP_HEADERS)
        assert r.status_code == 201
        jid = r.json()["job_id"]
        assert (tmp_path / "data" / "jobs" / jid / "capture.zip").read_bytes() == payload
        assert [p.name for p in (tmp_path / "data" / "jobs").iterdir()] == [jid]


def test_truncated_upload_is_400_and_nothing_left(tmp_path):
    with _client(tmp_path, _fast) as c:
        r = c.post("/api/v1/jobs", content=_multipart(b"abc", closing=False), headers=MP_HEADERS)
        assert r.status_code == 400
        assert list((tmp_path / "data" / "jobs").iterdir()) == []


def test_non_multipart_is_422(client):
    r = client.post("/api/v1/jobs", content=b"zip", headers={"content-type": "application/zip"})
    assert r.status_code == 422


def test_oversize_stream_stops_reading_at_cap(tmp_path):
    """Raw ASGI (TestClient buffers the whole body): the app must stop pulling body
    chunks once past the cap, i.e. nothing is spooled beyond it."""
    import anyio

    store = JobStore(tmp_path / "data", max_upload_bytes=100)
    app = create_app(store, Worker(store, _fast), PROFILES["laptop"])
    body = _multipart(b"x" * 200_000)
    chunks = [body[i : i + 1000] for i in range(0, len(body), 1000)]
    pulled = 0
    sent = []

    async def receive():
        nonlocal pulled
        pulled += 1
        i = pulled - 1
        return {"type": "http.request", "body": chunks[i], "more_body": i + 1 < len(chunks)}

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": "/api/v1/jobs", "raw_path": b"/api/v1/jobs", "query_string": b"",
        "root_path": "", "headers": [(b"content-type", MP_HEADERS["content-type"].encode())],
        "client": ("test", 1), "server": ("test", 80),
    }
    anyio.run(app, scope, receive, send)
    assert sent[0]["status"] == 413
    assert pulled < 100 < len(chunks)  # cap + 64 KiB form allowance ≈ 66 chunks
    assert list((tmp_path / "data" / "jobs").iterdir()) == []


def test_list_jobs(client):
    assert client.get("/api/v1/jobs").json() == {"jobs": []}
    jid = client.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}).json()["job_id"]
    wait_until(lambda: client.get(f"/api/v1/jobs/{jid}").json()["state"] == "done")
    jobs = client.get("/api/v1/jobs").json()["jobs"]
    assert len(jobs) == 1
    j = jobs[0]
    assert j["job_id"] == jid and j["state"] == "done" and j["progress"] == 1.0
    assert j["capture_bytes"] == 3 and j["has_scene"] is True
    assert isinstance(j["created_at"], float)


def test_list_jobs_includes_uploads_in_progress(client):
    data = b"u" * 4000
    uid = _begin(client, data)
    _put(client, uid, 0, data[:1000])
    jobs = client.get("/api/v1/jobs").json()["jobs"]
    assert len(jobs) == 1
    j = jobs[0]
    assert j["job_id"] == uid and j["state"] == "uploading" and j["progress"] == 0.25
    assert j["capture_bytes"] == 4000 and j["has_scene"] is False
    assert j["message"] == "已收到 1000 / 4000 字节"
    _put(client, uid, 1000, data[1000:])
    client.post(f"/api/v1/uploads/{uid}/finish")
    assert all(x["state"] != "uploading" for x in client.get("/api/v1/jobs").json()["jobs"])


def test_status_page_is_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "任务" in r.text and "/api/v1/jobs" in r.text


# --- resumable uploads (docs/superpowers/specs/2026-11-05-recon3d-resumable-transfer-design.md) ---

import hashlib


def _begin(c, data):
    r = c.post("/api/v1/uploads", json={"size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    assert r.status_code == 201 and r.json()["received"] == 0
    return r.json()["upload_id"]


def _put(c, uid, offset, body):
    return c.put(
        f"/api/v1/uploads/{uid}",
        content=body,
        headers={"Upload-Offset": str(offset), "Content-Type": "application/octet-stream"},
    )


def test_resumable_upload_flow(client):
    data = b"z" * 5000
    uid = _begin(client, data)
    assert _put(client, uid, 0, data[:2000]).json() == {"received": 2000}
    assert client.get(f"/api/v1/uploads/{uid}").json() == {"upload_id": uid, "size": 5000, "received": 2000}
    r = _put(client, uid, 1000, b"x")
    assert r.status_code == 409 and r.json() == {"detail": "偏移不匹配", "received": 2000}
    r = client.post(f"/api/v1/uploads/{uid}/finish")
    assert r.status_code == 409 and r.json() == {"detail": "上传未完成", "received": 2000}
    assert _put(client, uid, 2000, data[2000:]).json() == {"received": 5000}
    r = client.post(f"/api/v1/uploads/{uid}/finish")
    assert r.status_code == 201 and r.json() == {"job_id": uid}
    assert client.post(f"/api/v1/uploads/{uid}/finish").json() == {"job_id": uid}  # idempotent
    r = _put(client, uid, 5000, b"q")
    assert r.status_code == 409 and r.json()["detail"] == "上传已完成"
    assert client.get(f"/api/v1/uploads/{uid}").status_code == 404
    wait_until(lambda: client.get(f"/api/v1/jobs/{uid}").json()["state"] == "done")


def test_resumable_checksum_mismatch_is_422_and_gone(client):
    r = client.post("/api/v1/uploads", json={"size": 3, "sha256": "0" * 64})
    uid = r.json()["upload_id"]
    _put(client, uid, 0, b"abc")
    r = client.post(f"/api/v1/uploads/{uid}/finish")
    assert r.status_code == 422 and r.json()["detail"] == "文件校验失败，请重新上传"
    assert client.get(f"/api/v1/uploads/{uid}").status_code == 404


def test_resumable_bad_requests(tmp_path):
    with _client(tmp_path, _fast, max_upload_bytes=10) as c:
        assert c.post("/api/v1/uploads", json={"size": 11, "sha256": "0" * 64}).status_code == 413
        assert c.post("/api/v1/uploads", json={"size": 0, "sha256": "0" * 64}).status_code == 422
        assert c.post("/api/v1/uploads", json={"size": 5, "sha256": "nope"}).status_code == 422
        assert c.get("/api/v1/uploads/000000000000").status_code == 404
        assert c.get("/api/v1/uploads/000000000000").json() == {"detail": "上传不存在"}
        assert _put(c, "000000000000", 0, b"x").status_code == 404
        assert c.post("/api/v1/uploads/000000000000/finish").status_code == 404
        uid = c.post("/api/v1/uploads", json={"size": 4, "sha256": "0" * 64}).json()["upload_id"]
        assert _put(c, uid, 0, b"12345").status_code == 413
        assert c.get(f"/api/v1/uploads/{uid}").status_code == 404  # deleted with the 413
        uid = c.post("/api/v1/uploads", json={"size": 4, "sha256": "0" * 64}).json()["upload_id"]
        r = c.put(f"/api/v1/uploads/{uid}", content=b"1")
        assert r.status_code == 422 and r.json()["detail"] == "缺少 Upload-Offset"


def test_post_uploads_sweeps_expired(tmp_path):
    import json

    with _client(tmp_path, _fast) as c:
        old = c.post("/api/v1/uploads", json={"size": 4, "sha256": "0" * 64}).json()["upload_id"]
        meta = tmp_path / "data" / "jobs" / old / "upload.json"
        j = json.loads(meta.read_text(encoding="utf-8"))
        j["updated_at"] -= 90000
        meta.write_text(json.dumps(j), encoding="utf-8")
        c.post("/api/v1/uploads", json={"size": 4, "sha256": "0" * 64})
        assert c.get(f"/api/v1/uploads/{old}").status_code == 404


def test_put_disconnect_keeps_bytes(tmp_path):
    """Raw ASGI: the client vanishes after 3 of 6 body bytes; the 3 must be on disk and reported."""
    import anyio

    store = JobStore(tmp_path / "data")
    app = create_app(store, Worker(store, _fast), PROFILES["laptop"])
    info = store.begin_resumable(6, "0" * 64)
    sent = []

    async def run():
        pulled = 0

        async def receive():
            nonlocal pulled
            pulled += 1
            if pulled == 1:
                return {"type": "http.request", "body": b"123", "more_body": True}
            return {"type": "http.disconnect"}

        async def send(msg):
            sent.append(msg)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "PUT",
            "scheme": "http", "path": f"/api/v1/uploads/{info.upload_id}",
            "raw_path": f"/api/v1/uploads/{info.upload_id}".encode(), "query_string": b"",
            "root_path": "", "headers": [(b"upload-offset", b"0"), (b"content-type", b"application/octet-stream")],
            "client": ("test", 1), "server": ("test", 80),
        }
        await app(scope, receive, send)

    anyio.run(run)
    assert store.resumable(info.upload_id).received == 3


def test_concurrent_put_is_409(tmp_path):
    """A PUT while another is streaming in: 409 busy with the bytes on disk so far. The in-flight PUT
    is simulated by holding the store's per-upload lock (TestClient serialises request bodies, so a
    real overlapping request cannot be made through it); the lock itself is covered in test_jobs."""
    store = JobStore(tmp_path / "data")
    with TestClient(create_app(store, Worker(store, _fast), PROFILES["laptop"])) as c:
        uid = c.post("/api/v1/uploads", json={"size": 2, "sha256": "0" * 64}).json()["upload_id"]
        assert _put(c, uid, 0, b"1").json() == {"received": 1}
        with store._upload_lock(uid):
            r = _put(c, uid, 1, b"2")
            assert r.status_code == 409 and r.json() == {"detail": "该上传正在接收中", "received": 1}
        assert _put(c, uid, 1, b"2").json() == {"received": 2}


def _put_asgi(app, upload_id, offset, messages):
    """Raw ASGI PUT whose body arrives as several messages; returns (status, body dict)."""
    import anyio

    sent = []

    async def run():
        pulled = 0

        async def receive():
            nonlocal pulled
            if pulled < len(messages):
                body = messages[pulled]
                pulled += 1
                return {"type": "http.request", "body": body, "more_body": pulled < len(messages)}
            await anyio.sleep(3600)  # a client that never disconnects

        async def send(msg):
            sent.append(msg)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "PUT",
            "scheme": "http", "path": f"/api/v1/uploads/{upload_id}",
            "raw_path": f"/api/v1/uploads/{upload_id}".encode(), "query_string": b"",
            "root_path": "", "headers": [(b"upload-offset", str(offset).encode()), (b"content-type", b"application/octet-stream")],
            "client": ("test", 1), "server": ("test", 80),
        }
        with anyio.fail_after(10):
            await app(scope, receive, send)

    anyio.run(run)
    import json

    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return sent[0]["status"], json.loads(body or b"{}")


def test_busy_put_with_many_body_messages_still_answers_409(tmp_path):
    """uvicorn hands a 4 MB chunk over as dozens of messages; the 409 must come back without the
    handler stalling on a body nobody consumes (review finding C1)."""
    store = JobStore(tmp_path / "data")
    app = create_app(store, Worker(store, _fast), PROFILES["laptop"])
    info = store.begin_resumable(1000, "0" * 64)
    store.append_resumable(info.upload_id, 0, [b"1"])
    with store._upload_lock(info.upload_id):
        status, body = _put_asgi(app, info.upload_id, 1, [b"x" * 10] * 12)
    assert status == 409 and body == {"detail": "该上传正在接收中", "received": 1}
    status, body = _put_asgi(app, info.upload_id, 0, [b"x" * 10] * 12)
    assert status == 409 and body == {"detail": "偏移不匹配", "received": 1}


def test_oversize_put_with_many_body_messages_is_413(tmp_path):
    store = JobStore(tmp_path / "data")
    app = create_app(store, Worker(store, _fast), PROFILES["laptop"])
    info = store.begin_resumable(15, "0" * 64)
    status, body = _put_asgi(app, info.upload_id, 0, [b"x" * 10] * 12)
    assert status == 413
    assert store.resumable(info.upload_id) is None


# --- delete (docs/superpowers/specs/2026-11-05-recon3d-delete-design.md) ---


def test_delete_job(client):
    jid = client.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}).json()["job_id"]
    wait_until(lambda: client.get(f"/api/v1/jobs/{jid}").json()["state"] == "done")
    r = client.delete(f"/api/v1/jobs/{jid}")
    assert r.status_code == 204 and r.content == b""
    assert client.get(f"/api/v1/jobs/{jid}").status_code == 404
    assert client.delete(f"/api/v1/jobs/{jid}").status_code == 404
    assert client.delete("/api/v1/jobs/zzz").json() == {"detail": "任务不存在"}


def test_delete_training_job_is_409(client_slow):
    jid = client_slow.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}).json()["job_id"]
    wait_until(lambda: client_slow.get(f"/api/v1/jobs/{jid}").json()["state"] == "training")
    r = client_slow.delete(f"/api/v1/jobs/{jid}")
    assert r.status_code == 409 and r.json() == {"detail": "任务正在训练，完成后再删"}
    assert client_slow.get(f"/api/v1/jobs/{jid}").status_code == 200


def test_delete_resumable_upload(tmp_path):
    store = JobStore(tmp_path / "data")
    with TestClient(create_app(store, Worker(store, _fast), PROFILES["laptop"])) as c:
        data = b"d" * 3000
        uid = _begin(c, data)
        _put(c, uid, 0, data[:1000])
        with store._upload_lock(uid):
            r = c.delete(f"/api/v1/jobs/{uid}")
            assert r.status_code == 409 and r.json() == {"detail": "该上传正在接收中", "received": 1000}
        assert c.delete(f"/api/v1/jobs/{uid}").status_code == 204
        assert c.get(f"/api/v1/uploads/{uid}").status_code == 404
        assert all(j["job_id"] != uid for j in c.get("/api/v1/jobs").json()["jobs"])
