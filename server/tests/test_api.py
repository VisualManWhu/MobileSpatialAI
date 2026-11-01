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
