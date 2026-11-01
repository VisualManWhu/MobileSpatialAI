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
