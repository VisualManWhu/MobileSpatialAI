# Resumable Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Uploads and downloads between the phone app and the compute server survive connection drops and app restarts by resuming from the last byte the other side confirmed.

**Architecture:** The server gets a tus-style resumable upload (`POST /uploads`, `PUT /uploads/{id}` with `Upload-Offset`, `GET /uploads/{id}`, `POST /uploads/{id}/finish`) stored as `data/jobs/<id>/capture.zip.part` + `upload.json`; `finish` turns the directory into an ordinary job through the existing `_commit`. Downloads need nothing new on the server (Starlette 1.7 `FileResponse` already answers `Range`). The Android `ApiClient` drives the chunk loop with reconnect/backoff and resumes downloads with `Range`; `UploadViewModel` persists the upload id in `rooms.json` so a killed app continues where it was.

**Tech Stack:** Python 3.12 / FastAPI 0.162 / Starlette 1.7 (server, run in WSL with `uv`); Kotlin / OkHttp 4.12 / kotlinx.serialization / MockWebServer (Android).

**Spec:** `docs/superpowers/specs/2026-11-05-recon3d-resumable-transfer-design.md`

## Global Constraints

- User-visible strings (error `detail`, status messages, button labels) are Chinese; code and comments are English.
- Every error body stays `{"detail": "<中文>"}`; extra keys (`received`) are additive.
- `upload_id` == later `job_id`: 12 hex chars from `uuid4().hex[:12]`, directory `data/jobs/<id>/`.
- Chunk size on the phone: 4 MB. Retry: backoff `2 s × 2^(attempt−1)` capped at 30 s, give up after 30 consecutive failures; a successful PUT/GET resets the count.
- Expiry: uploads whose `updated_at` is older than 24 h are deleted at startup and on each `POST /uploads`; an upload whose lock is held is skipped.
- Server PUT idle timeout: 30 s without body data counts as a disconnect (bytes kept, lock released).
- Client error classes: `IOException`, 5xx and `409 该上传正在接收中` are retryable; `409 偏移不匹配` corrects the offset; `404` on `GET /uploads/{id}` falls back to `GET /jobs/{id}` (200 → that id is the job) before starting over; any other 4xx is fatal and clears `uploadId`.
- Old `POST /jobs` multipart endpoint stays as is.
- Server commands run in WSL: `wsl -d Ubuntu -- bash -lc 'cd /mnt/d/projects/Recon3D/server && export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d UV_LINK_MODE=copy && uv run pytest …'`. Android unit tests: `cd android; .\gradlew.bat :app:testDebugUnitTest --tests '<class>'` (PowerShell).
- Python files are written with `newline="\n"`. Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A PUT whose body is cut mid-stream: bytes written so far stay, `received` reflects the file size, the next `GET /uploads/{id}` reports it (Task 2 `test_put_disconnect_keeps_bytes`).
2. Server killed between a PUT and the `upload.json` write: on startup `received` is corrected from the `.part` size (Task 1 `test_startup_corrects_received_from_part_size`).
3. Two PUTs for the same id at once: the second gets 409 and the file is not interleaved (Task 2 `test_concurrent_put_is_409`).
4. Phone's `uploadId` points at an upload the server already expired: client restarts from a fresh `POST /uploads` and calls `onCreated` again (Task 3 `uploadRestartsWhenServerForgotTheUpload`).
5. Download `.part` left by an older server build that does not honour `Range` (plain 200): the client overwrites from the start instead of appending (Task 3 `downloadPlain200OverwritesPart`).

---

### Task 1: Resumable upload storage (`jobs/store.py`)

**Files:**
- Modify: `server/recon3d/jobs/store.py`
- Test: `server/tests/test_jobs.py` (append)

**Interfaces:**
- Produces on `JobStore`:
  - `begin_resumable(size: int, sha256: str) -> ResumableInfo` — raises `UploadTooLargeError` when `size > max_upload_bytes`, `ValueError` when `size <= 0` or sha256 is not 64 lowercase hex.
  - `resumable(upload_id: str) -> ResumableInfo | None` — None when the id is malformed, unknown, or already finished (no `upload.json`).
  - `append_resumable(upload_id: str, offset: int, chunks: Iterable[bytes]) -> int` — writes chunks at `offset` (must equal `received`, else `OffsetMismatch(received)`), returns new `received`. Raises `UploadTooLargeError` and deletes the upload when the total would exceed `size`. Writes partial data even if the iterable raises (caller's disconnect): update `upload.json` in a `finally`. Holds a per-id `threading.Lock`; a second concurrent call raises `UploadBusy(received)`.
  - `finish_resumable(upload_id: str) -> str` — `received != size` → `IncompleteUpload(received)`; SHA-256 mismatch → `ChecksumMismatch` and the directory is deleted; otherwise rename `.part` → `capture.zip`, delete `upload.json`, `_commit(id, size)`, return id. If the id is already a job (has `status.json`, no `upload.json`) return it unchanged (idempotent).
  - `sweep_resumable(now: float | None = None, max_age_s: float = 86400) -> int` — deletes expired upload dirs whose lock is not held, returns count. `recover_after_restart` calls it and also corrects `received` from the `.part` size for the remaining ones.
  - `@dataclass(frozen=True) ResumableInfo(upload_id: str, size: int, sha256: str, received: int, created_at: float, updated_at: float)`.
  - Exceptions in `store.py`: `OffsetMismatch(received)`, `UploadBusy(received)`, `IncompleteUpload(received)` (all with `.received`), `ChecksumMismatch`.

- [ ] **Step 1: Write the failing tests** (append to `server/tests/test_jobs.py`)

```python
import hashlib, time
import pytest
from recon3d.jobs.store import (JobStore, UploadTooLargeError, OffsetMismatch, UploadBusy,
                                IncompleteUpload, ChecksumMismatch)

def _store(tmp_path, **kw): return JobStore(tmp_path / "data", **kw)

def test_resumable_roundtrip(tmp_path):
    s = _store(tmp_path); data = b"abc" * 1000
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

def test_resumable_rejects_bad_begin(tmp_path):
    s = _store(tmp_path, max_upload_bytes=10)
    with pytest.raises(UploadTooLargeError): s.begin_resumable(11, "0" * 64)
    with pytest.raises(ValueError): s.begin_resumable(0, "0" * 64)
    with pytest.raises(ValueError): s.begin_resumable(5, "xyz")

def test_resumable_offset_mismatch_reports_received(tmp_path):
    s = _store(tmp_path); i = s.begin_resumable(10, "0" * 64)
    s.append_resumable(i.upload_id, 0, [b"12345"])
    with pytest.raises(OffsetMismatch) as e: s.append_resumable(i.upload_id, 3, [b"x"])
    assert e.value.received == 5

def test_resumable_partial_iterable_keeps_bytes(tmp_path):
    s = _store(tmp_path); i = s.begin_resumable(10, "0" * 64)
    def chunks():
        yield b"1234"; raise ConnectionError
    with pytest.raises(ConnectionError): s.append_resumable(i.upload_id, 0, chunks())
    assert s.resumable(i.upload_id).received == 4

def test_resumable_over_declared_size_deletes(tmp_path):
    s = _store(tmp_path); i = s.begin_resumable(4, "0" * 64)
    with pytest.raises(UploadTooLargeError): s.append_resumable(i.upload_id, 0, [b"12345"])
    assert s.resumable(i.upload_id) is None and not s.job_dir(i.upload_id).exists()

def test_resumable_finish_incomplete_and_checksum(tmp_path):
    s = _store(tmp_path); i = s.begin_resumable(4, "0" * 64)
    s.append_resumable(i.upload_id, 0, [b"12"])
    with pytest.raises(IncompleteUpload) as e: s.finish_resumable(i.upload_id)
    assert e.value.received == 2
    s.append_resumable(i.upload_id, 2, [b"34"])
    with pytest.raises(ChecksumMismatch): s.finish_resumable(i.upload_id)
    assert not s.job_dir(i.upload_id).exists()

def test_resumable_busy(tmp_path):
    import threading
    s = _store(tmp_path); i = s.begin_resumable(10, "0" * 64)
    started, release = threading.Event(), threading.Event()
    def slow():
        yield b"1"; started.set(); release.wait(5); yield b"2"
    t = threading.Thread(target=lambda: s.append_resumable(i.upload_id, 0, slow())); t.start()
    started.wait(5)
    with pytest.raises(UploadBusy): s.append_resumable(i.upload_id, 1, [b"x"])
    release.set(); t.join(5)
    assert s.resumable(i.upload_id).received == 2

def test_sweep_and_startup_correct_received_from_part_size(tmp_path):
    s = _store(tmp_path)
    old = s.begin_resumable(10, "0" * 64); fresh = s.begin_resumable(10, "0" * 64)
    meta = s.job_dir(old.upload_id) / "upload.json"
    j = json.loads(meta.read_text()); j["updated_at"] -= 90000; meta.write_text(json.dumps(j))
    # simulate a crash after bytes landed but before upload.json was updated
    (s.job_dir(fresh.upload_id) / "capture.zip.part").write_bytes(b"123")
    assert s.sweep_resumable() == 1
    assert s.resumable(old.upload_id) is None
    s.recover_after_restart()
    assert s.resumable(fresh.upload_id).received == 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_jobs.py -q -k resumable`
Expected: FAIL with `ImportError` (names not defined).

- [ ] **Step 3: Implement in `server/recon3d/jobs/store.py`**

`upload.json` holds `{"size", "sha256", "received", "created_at", "updated_at"}` written atomically (reuse the `_write`-style mkstemp + `os.replace`). `append_resumable` opens the `.part` in `"r+b"`, seeks to `offset`, writes chunks, tracks bytes, and in `finally` truncates nothing, `fsync`s, then stores `received = file size`. Per-id locks: `dict[str, threading.Lock]` guarded by `self._lock`; acquire with `blocking=False`. `finish_resumable` streams the `.part` through `hashlib.sha256` (1 MB blocks). `recover_after_restart` first calls `sweep_resumable()`, then for every remaining `upload.json` sets `received` to the `.part` size.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_jobs.py -q`
Expected: all pass (existing tests included).

- [ ] **Step 5: Commit**

```bash
git add server/recon3d/jobs/store.py server/tests/test_jobs.py
git commit -m "feat(server): resumable upload storage in JobStore"
```

---

### Task 2: Resumable upload endpoints + `docs/api.md`

**Files:**
- Create: `server/recon3d/api/resumable.py`
- Modify: `server/recon3d/api/app.py` (register router)
- Modify: `docs/api.md`
- Test: `server/tests/test_api.py` (append)

**Interfaces:**
- Consumes Task 1's `JobStore` methods and exceptions.
- Produces: `def router(store: JobStore) -> APIRouter` in `resumable.py`, included by `create_app` under `/api/v1` (router has prefix `/uploads`). Response bodies exactly as spec §3. Chinese details: `"偏移不匹配"`, `"该上传正在接收中"`, `"上传已完成"`, `"上传不存在"`, `"上传未完成"`, `"文件校验失败，请重新上传"`; 413 reuses `upload.too_large(store)`; 422 for bad body uses FastAPI's pydantic validation (model `UploadCreate(size: int = Field(gt=0), sha256: str = Field(pattern=r"^[0-9a-f]{64}$"))`).

- [ ] **Step 1: Write the failing tests** (append to `server/tests/test_api.py`)

```python
import hashlib

def _begin(c, data):
    r = c.post("/api/v1/uploads", json={"size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    assert r.status_code == 201 and r.json()["received"] == 0
    return r.json()["upload_id"]

def _put(c, uid, offset, body):
    return c.put(f"/api/v1/uploads/{uid}", content=body,
                 headers={"Upload-Offset": str(offset), "Content-Type": "application/octet-stream"})

def test_resumable_upload_flow(client):
    data = b"z" * 5000; uid = _begin(client, data)
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
    assert _put(client, uid, 5000, b"q").status_code == 409
    assert client.get(f"/api/v1/uploads/{uid}").status_code == 404
    wait_until(lambda: client.get(f"/api/v1/jobs/{uid}").json()["state"] == "done")

def test_resumable_checksum_mismatch_is_422_and_gone(client):
    r = client.post("/api/v1/uploads", json={"size": 3, "sha256": "0" * 64}); uid = r.json()["upload_id"]
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
        assert _put(c, "000000000000", 0, b"x").status_code == 404
        uid = c.post("/api/v1/uploads", json={"size": 4, "sha256": "0" * 64}).json()["upload_id"]
        assert _put(c, uid, 0, b"12345").status_code == 413
        assert c.get(f"/api/v1/uploads/{uid}").status_code == 404
        assert _put(c, uid, 0, b"1").status_code == 404  # missing Upload-Offset → 422 checked below
        uid = c.post("/api/v1/uploads", json={"size": 4, "sha256": "0" * 64}).json()["upload_id"]
        assert c.put(f"/api/v1/uploads/{uid}", content=b"1").status_code == 422

def test_put_disconnect_keeps_bytes(tmp_path):
    """Raw ASGI: the client vanishes after 3 of 6 body bytes; the 3 must be on disk and reported."""
    import anyio
    store = JobStore(tmp_path / "data")
    app = create_app(store, Worker(store, _fast), PROFILES["laptop"])
    info = store.begin_resumable(6, "0" * 64)
    async def run():
        sent = 0
        async def receive():
            nonlocal sent
            if sent == 0:
                sent = 1; return {"type": "http.request", "body": b"123", "more_body": True}
            return {"type": "http.disconnect"}
        async def send(msg): pass
        scope = {"type": "http", "method": "PUT", "path": f"/api/v1/uploads/{info.upload_id}",
                 "headers": [(b"upload-offset", b"0"), (b"content-type", b"application/octet-stream")],
                 "query_string": b"", "root_path": "", "scheme": "http", "server": ("t", 80), "client": ("c", 1)}
        await app(scope, receive, send)
    anyio.run(run)
    assert store.resumable(info.upload_id).received == 3

def test_concurrent_put_is_409(client_slow): ...
```

For `test_concurrent_put_is_409`: start a thread doing a PUT whose body is a generator yielding `b"1"`, setting an `Event`, then blocking on a second `Event`; after the first event, a second PUT with offset 1 must return 409 with `detail == "该上传正在接收中"` and `received == 1`; release, join, final `received == 2`. (`TestClient` accepts a generator as `content`.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_api.py -q -k "resumable or put_"`
Expected: 404s / FAIL.

- [ ] **Step 3: Implement `server/recon3d/api/resumable.py` and register it**

PUT handler: read `Upload-Offset` (missing/non-int → 422 `"缺少 Upload-Offset"`); wrap each `request.stream()` step in `anyio.fail_after(30)` (`PUT_IDLE_S = 30`) and treat `TimeoutError` like a disconnect; then run `store.append_resumable(uid, offset, chunks)` in a worker thread where `chunks` is fed from `request.stream()` through an `anyio` memory channel (or iterate `request.stream()` in the async handler and push into a thread-run writer via `to_thread.run_sync` per chunk, as `upload.py` does with the multipart parser — pick the latter for symmetry: open file once in a small `_Writer` object, call `writer.write(chunk)` via `run_sync`, `writer.close()` in `finally`). Mapping: `OffsetMismatch` → 409 `{"detail":"偏移不匹配","received":n}` (use `JSONResponse`, not `HTTPException`, so the extra key survives); `UploadBusy` → 409 `"该上传正在接收中"`; `UploadTooLargeError` → 413 via `too_large`; unknown id → 404 `"上传不存在"`; id that is already a job → 409 `"上传已完成"`. `POST /uploads` calls `store.sweep_resumable()` first. `finish` maps `IncompleteUpload` → 409 with `received`, `ChecksumMismatch` → 422.

- [ ] **Step 4: Run all server tests**

Run: `uv run pytest -q -m "not gpu"`
Expected: all pass.

- [ ] **Step 5: Document in `docs/api.md`**

Add a section `## 断点续传上传（手机用）` with the four endpoints, request/response bodies and error table exactly as spec §3; mark `## POST /jobs` as `旧接口（命令行 / 测试用，手机不再使用）`; note under `GET /jobs/{id}/scene.splat` that `Range` is honoured (`206`, `Accept-Ranges: bytes`).

- [ ] **Step 6: Commit**

```bash
git add server/recon3d/api/resumable.py server/recon3d/api/app.py server/tests/test_api.py docs/api.md
git commit -m "feat(server): resumable upload endpoints (/api/v1/uploads)"
```

---

### Task 3: Android `ApiClient` resumable upload and ranged download

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/net/ApiClient.kt`
- Modify: `android/app/src/main/java/dev/recon3d/net/Models.kt`
- Modify: `android/app/src/main/java/dev/recon3d/AppGraph.kt` (`SettingsBackedApi` forwards the new signatures)
- Test: `android/app/src/test/java/dev/recon3d/net/ApiClientTest.kt`

**Interfaces:**
- Produces in `Api`:
  ```kotlin
  interface UploadListener {
      fun onCreated(uploadId: String)
      fun onProgress(fraction: Float)
      fun onRetry(attempt: Int)
  }
  suspend fun upload(zip: File, uploadId: String?, listener: UploadListener): String
  suspend fun downloadScene(id: String, dest: File, onRetry: (Int) -> Unit = {})
  ```
  (the old `upload(zip, onProgress)` is removed; Task 4 updates callers and the test fake.)
- Models: `@Serializable internal data class UploadCreated(@SerialName("upload_id") val uploadId: String, val received: Long)`, `UploadInfo(@SerialName("upload_id") val uploadId: String, val size: Long, val received: Long)`, `Received(val received: Long)`.
- Constants on `ApiClient.Companion`: `CHUNK_BYTES = 4L * 1024 * 1024`, `MAX_TRANSFER_RETRIES = 30`, `MSG_CHECKSUM = "文件校验失败，请重新上传"` is not a constant: it is whatever `detail` the server returns, surfaced through the existing `httpError`. A `ChecksumFailed` marker is not needed: Task 4 matches on the 422 detail text through a new `ApiException.code: Int?` (HTTP status, null for transport errors) — add that field now.
- Backoff is injectable for tests: constructor param `private val sleep: suspend (Long) -> Unit = { delay(it) }`.

- [ ] **Step 1: Write the failing tests** (in `ApiClientTest`; helper `noSleep` client built with `sleep = {}`)

```kotlin
private class RecordingListener : UploadListener {
    val created = mutableListOf<String>(); val progress = mutableListOf<Float>(); val retries = mutableListOf<Int>()
    override fun onCreated(uploadId: String) { created += uploadId }
    override fun onProgress(fraction: Float) { progress += fraction }
    override fun onRetry(attempt: Int) { retries += attempt }
}

@Test fun uploadCreatesPutsChunksAndFinishes() = runTest {
    val zip = tmpZip(10 * 1024 * 1024 + 5)  // 2 full chunks + 5 bytes
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"upload_id":"abcdefabcdef","received":0}"""))
    server.enqueue(MockResponse().setBody("""{"received":4194304}"""))
    server.enqueue(MockResponse().setBody("""{"received":8388608}"""))
    server.enqueue(MockResponse().setBody("""{"received":10485765}"""))
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"abcdefabcdef"}"""))
    val l = RecordingListener()
    assertEquals("abcdefabcdef", client.upload(zip, null, l))
    val create = server.takeRequest()
    assertEquals("POST", create.method); assertEquals("/api/v1/uploads", create.path)
    val body = Json.parseToJsonElement(create.body.readUtf8()).jsonObject
    assertEquals(zip.length(), body["size"]!!.jsonPrimitive.long)
    assertEquals(sha256Hex(zip), body["sha256"]!!.jsonPrimitive.content)
    val put1 = server.takeRequest()
    assertEquals("PUT", put1.method); assertEquals("/api/v1/uploads/abcdefabcdef", put1.path)
    assertEquals("0", put1.getHeader("Upload-Offset")); assertEquals(4194304L, put1.bodySize)
    assertEquals("4194304", server.takeRequest().getHeader("Upload-Offset"))
    val put3 = server.takeRequest(); assertEquals("8388608", put3.getHeader("Upload-Offset")); assertEquals(5L, put3.bodySize)
    assertEquals("/api/v1/uploads/abcdefabcdef/finish", server.takeRequest().path)
    assertEquals(listOf("abcdefabcdef"), l.created)
    assertEquals(1f, l.progress.last()); assertTrue(l.retries.isEmpty())
}

@Test fun uploadResumesFromServerReceivedAndCorrectsOffsetOn409() = runTest {
    val zip = tmpZip(6 * 1024 * 1024)
    server.enqueue(MockResponse().setBody("""{"upload_id":"abcdefabcdef","size":6291456,"received":1000}"""))
    server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"偏移不匹配","received":2000}"""))
    server.enqueue(MockResponse().setBody("""{"received":6291456}"""))
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"abcdefabcdef"}"""))
    val l = RecordingListener()
    client.upload(zip, "abcdefabcdef", l)
    assertEquals("GET", server.takeRequest().method.also { })  // GET /uploads/{id}
    assertEquals("1000", server.takeRequest().getHeader("Upload-Offset"))
    val put2 = server.takeRequest(); assertEquals("2000", put2.getHeader("Upload-Offset")); assertEquals(6291456L - 2000, put2.bodySize)
    assertTrue(l.created.isEmpty()); assertTrue(l.retries.isEmpty())
}

@Test fun uploadRetriesAfterDisconnectThenContinues() = runTest {
    val zip = tmpZip(100)
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"upload_id":"abcdefabcdef","received":0}"""))
    server.enqueue(MockResponse().setSocketPolicy(SocketPolicy.DISCONNECT_DURING_REQUEST_BODY))
    server.enqueue(MockResponse().setBody("""{"upload_id":"abcdefabcdef","size":100,"received":40}"""))
    server.enqueue(MockResponse().setBody("""{"received":100}"""))
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"abcdefabcdef"}"""))
    val l = RecordingListener()
    noSleep.upload(zip, null, l)
    assertEquals(listOf(1), l.retries)
    server.takeRequest(); server.takeRequest()
    assertEquals("GET", server.takeRequest().method)
    assertEquals("40", server.takeRequest().getHeader("Upload-Offset"))
}

@Test fun uploadGivesUpAfterThirtyConsecutiveFailures() {
    val zip = tmpZip(10)
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"upload_id":"abcdefabcdef","received":0}"""))
    repeat(31) { server.enqueue(MockResponse().setSocketPolicy(SocketPolicy.DISCONNECT_AT_START)) }
    val l = RecordingListener()
    val e = apiError { noSleep.upload(zip, null, l) }
    assertEquals(ApiClient.MSG_UNREACHABLE, e.messageZh); assertEquals(30, l.retries.size)
}

@Test fun uploadFinishedBeforeKillReturnsExistingJob() = runTest {
    val zip = tmpZip(10)
    server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"上传不存在"}"""))
    server.enqueue(MockResponse().setBody("""{"job_id":"000000000000","state":"training","progress":0.1,"message":"训练中"}"""))
    assertEquals("000000000000", client.upload(zip, "000000000000", RecordingListener()))
    server.takeRequest(); assertEquals("/api/v1/jobs/000000000000", server.takeRequest().path)
    assertEquals(2, server.requestCount)
}

@Test fun uploadFatal4xxCarriesCodeWithoutRetry() {
    val zip = tmpZip(3)
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"upload_id":"abcdefabcdef","received":0}"""))
    server.enqueue(MockResponse().setResponseCode(413).setBody("""{"detail":"上传文件过大，上限 4 GB"}"""))
    val l = RecordingListener()
    val e = apiError { noSleep.upload(zip, null, l) }
    assertEquals(413, e.code); assertTrue(l.retries.isEmpty()); assertEquals(2, server.requestCount)
}

@Test fun uploadBusy409IsRetriedWithBackoff() = runTest {
    val zip = tmpZip(3)
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"upload_id":"abcdefabcdef","received":0}"""))
    server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"该上传正在接收中","received":0}"""))
    server.enqueue(MockResponse().setBody("""{"upload_id":"abcdefabcdef","size":3,"received":0}"""))
    server.enqueue(MockResponse().setBody("""{"received":3}"""))
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"abcdefabcdef"}"""))
    val l = RecordingListener()
    noSleep.upload(zip, null, l)
    assertEquals(listOf(1), l.retries)
}

@Test fun uploadRestartsWhenServerForgotTheUpload() = runTest {
    val zip = tmpZip(10)
    server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"上传不存在"}"""))
    server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"任务不存在"}"""))
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"upload_id":"111111111111","received":0}"""))
    server.enqueue(MockResponse().setBody("""{"received":10}"""))
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"111111111111"}"""))
    val l = RecordingListener()
    assertEquals("111111111111", client.upload(zip, "000000000000", l))
    assertEquals(listOf("111111111111"), l.created)
}

@Test fun uploadChecksumFailureCarriesServerDetailAndCode() {
    val zip = tmpZip(3)
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"upload_id":"abcdefabcdef","received":0}"""))
    server.enqueue(MockResponse().setBody("""{"received":3}"""))
    server.enqueue(MockResponse().setResponseCode(422).setBody("""{"detail":"文件校验失败，请重新上传"}"""))
    val e = apiError { client.upload(zip, null, RecordingListener()) }
    assertEquals("文件校验失败，请重新上传", e.messageZh); assertEquals(422, e.code)
}

@Test fun downloadResumesPartWithRangeAnd206() = runTest {
    val dest = File(tmp.root, "s/scene.splat"); dest.parentFile!!.mkdirs()
    File(dest.path + ".part").writeBytes(byteArrayOf(1, 2))
    server.enqueue(MockResponse().setResponseCode(206).setHeader("Content-Range", "bytes 2-3/4").setBody(Buffer().write(byteArrayOf(3, 4))))
    client.downloadScene("j", dest)
    assertEquals("bytes=2-", server.takeRequest().getHeader("Range"))
    assertArrayEquals(byteArrayOf(1, 2, 3, 4), dest.readBytes()); assertFalse(File(dest.path + ".part").exists())
}

@Test fun downloadPlain200OverwritesPart() = runTest {
    val dest = File(tmp.root, "s/scene.splat"); dest.parentFile!!.mkdirs()
    File(dest.path + ".part").writeBytes(byteArrayOf(9, 9))
    server.enqueue(MockResponse().setBody(Buffer().write(byteArrayOf(1, 2, 3))))
    client.downloadScene("j", dest)
    assertArrayEquals(byteArrayOf(1, 2, 3), dest.readBytes())
}

@Test fun downloadRetriesAfterDropAndResumes() = runTest {
    val dest = File(tmp.root, "s/scene.splat")
    server.enqueue(MockResponse().setBody(Buffer().write(byteArrayOf(1, 2, 3, 4))).setSocketPolicy(SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY))
    server.enqueue(MockResponse().setResponseCode(206).setHeader("Content-Range", "bytes 2-3/4").setBody(Buffer().write(byteArrayOf(3, 4))))
    val retries = mutableListOf<Int>()
    noSleep.downloadScene("j", dest) { retries += it }
    assertEquals(listOf(1), retries)
    assertArrayEquals(byteArrayOf(1, 2, 3, 4), dest.readBytes())
}
```

`DISCONNECT_DURING_RESPONSE_BODY` sends half the body, so the first attempt leaves a 2-byte `.part`; the test must assert the second request's `Range` is `bytes=2-` (take both requests). Delete `uploadSendsMultipartFileToJobs` and `uploadReportsMonotonicProgressEndingAtOne`; keep `uploadTooLargeUsesServerDetail` by pointing it at the 413 from `POST /uploads`.

- [ ] **Step 2: Run to verify they fail**

Run: `.\gradlew.bat :app:testDebugUnitTest --tests 'dev.recon3d.net.ApiClientTest'`
Expected: compile errors (new API).

- [ ] **Step 3: Implement**

In `ApiClient`: `sha256Hex(file: File): String` (internal top-level, 1 MB buffer); `private class FileRangeBody(file, offset, length): RequestBody` streaming with `FileInputStream.skip` + `okio.source().buffer()` limited to `length` (`contentLength() = length`); `upload` as spec §4 steps 1–5; the transfer retry loop as a private `suspend fun <T> withTransferRetry(onRetry: (Int) -> Unit, block: suspend () -> T): T` that catches `IOException` and `ApiException` whose `code` is `null`, `>= 500`, or 409 with detail `该上传正在接收中`, sleeps `minOf(2_000L shl (attempt - 1), 30_000L)` ms via `sleep`, and rethrows as `ApiException(MSG_UNREACHABLE)` after `MAX_TRANSFER_RETRIES`. 409 with a `received` key is handled inside the loop body (parse, set offset, continue) without counting as a failure. `downloadScene`: `Range` header when `.part` exists; 206 → append, 200 → truncate, 416 → delete `.part` and retry once from scratch. `ApiException` gains `val code: Int? = null`; `httpError` passes `response.code`.

- [ ] **Step 4: Run to verify they pass**

Run: `.\gradlew.bat :app:testDebugUnitTest --tests 'dev.recon3d.net.ApiClientTest'`
Expected: BUILD SUCCESSFUL (UploadViewModelTest will not compile yet — run only this class; Task 4 fixes it).

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/dev/recon3d/net android/app/src/main/java/dev/recon3d/AppGraph.kt android/app/src/test/java/dev/recon3d/net/ApiClientTest.kt
git commit -m "feat(app): resumable chunked upload and ranged download in ApiClient"
```

---

### Task 4: `UploadViewModel` persistence of the upload id and reconnect UI

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/rooms/RoomRepository.kt` (`CaptureRecord.uploadId: String? = null`)
- Modify: `android/app/src/main/java/dev/recon3d/upload/UploadScreen.kt`
- Test: `android/app/src/test/java/dev/recon3d/upload/UploadViewModelTest.kt`

**Interfaces:**
- Consumes Task 3's `Api.upload(zip, uploadId, listener)`, `Api.downloadScene(id, dest, onRetry)`, `ApiException.code`.
- Produces: `const val MSG_RECONNECTING_PREFIX = "网络中断，正在重连"`, `internal fun reconnectingMessage(n: Int) = "$MSG_RECONNECTING_PREFIX ($n)"`; `captureStatusText` renders `上传中 37% · 网络中断，正在重连 (3)` and, while `exporting` with `MSG_DOWNLOADING`, `正在下载模型… · 网络中断，正在重连 (n)`; `retryButtonLabel`: `jobId != null` → `"重试"`, else `uploadId != null` → `"继续上传"`, else `"重新上传"`.

- [ ] **Step 1: Update `FakeApi` and write the failing tests**

`FakeApi.upload(zip, uploadId, listener)`: records `lastUploadId`, calls `listener.onCreated("u1")` when `uploadId == null`, replays `retrySteps: List<Int>` through `onRetry`, then `progressSteps` through `onProgress`, returns `"j"`. `downloadScene(id, dest, onRetry)` replays `downloadRetrySteps`.

```kotlin
@Test fun uploadIdIsPersistedThenClearedOnJobCreation() = runTest {
    val vm = vm(); api.progressSteps = listOf(0.3f); api.uploadDelayMs = 100
    vm.start(room, cap); advanceTimeBy(50); runCurrent()
    assertEquals("u1", repo.capture(room, cap).uploadId)
    advanceUntilIdle()
    val rec = repo.capture(room, cap); assertNull(rec.uploadId); assertEquals("j", rec.jobId)
}

@Test fun resumeInterruptedContinuesUploadWithStoredId() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "uploading", uploadId = "u9", progress = 0.4f) }
    vm().resumeInterrupted(); advanceUntilIdle()
    assertEquals("u9", api.lastUploadId); assertEquals("done", repo.capture(room, cap).state)
}

@Test fun resumeInterruptedWithoutIdStillFails() = runTest {  // old-version records
    repo.editCapture(room, cap) { it.copy(state = "uploading") }
    vm().resumeInterrupted(); advanceUntilIdle()
    assertEquals("failed", repo.capture(room, cap).state); assertEquals(MSG_UPLOAD_INTERRUPTED, repo.capture(room, cap).message)
}

@Test fun reconnectMessageShownDuringUploadAndClearedAfter() = runTest {
    api.retrySteps = listOf(1, 2); api.uploadDelayMs = 100
    val vm = vm(); vm.start(room, cap); advanceTimeBy(50); runCurrent()
    assertEquals(reconnectingMessage(2), repo.capture(room, cap).message)
    assertEquals("上传中 0% · 网络中断，正在重连 (2)", captureStatusText(repo.capture(room, cap)))
    advanceUntilIdle(); assertNull(repo.capture(room, cap).message?.takeIf { it.startsWith(MSG_RECONNECTING_PREFIX) })
}

@Test fun checksumFailureClearsUploadIdAndOffersReupload() = runTest {
    api.failUploadOnce(ApiException("文件校验失败，请重新上传", code = 422))
    val vm = vm(); vm.start(room, cap); advanceUntilIdle()
    val rec = repo.capture(room, cap)
    assertEquals("failed", rec.state); assertNull(rec.uploadId); assertEquals("重新上传", retryButtonLabel(rec))
}

@Test fun transportFailureKeepsUploadIdAndOffersContinue() = runTest {
    api.failUploadOnce(ApiException(ApiClient.MSG_UNREACHABLE))  // after onCreated
    ...assertEquals("u1", rec.uploadId); assertEquals("继续上传", retryButtonLabel(rec))
}
```

(For the last test, `FakeApi` calls `onCreated` before throwing the scripted failure.) Adjust the existing `uploadProgressPersistedOnlyOnWholePercentChange` and `concurrentStartsUploadOnce` to the new fake signature.

- [ ] **Step 2: Run to verify they fail**

Run: `.\gradlew.bat :app:testDebugUnitTest --tests 'dev.recon3d.upload.UploadViewModelTest'`
Expected: compile errors / failures.

- [ ] **Step 3: Implement in `UploadScreen.kt` and `RoomRepository.kt`**

`upload()`: pass `repo.capture(...).uploadId`; listener writes `uploadId` on `onCreated`, `progress` on whole-percent change (existing throttle), `message = reconnectingMessage(n)` on `onRetry`, and clears a reconnecting message on the next `onProgress`. After `api.upload` returns: `state = "queued", jobId, uploadId = null`. Catch in `launchOnce`: `ApiException` with `code in 400..499` during upload → `fail(..., clearUploadId = true)`; transport failure (`code == null`) keeps `uploadId`. `captureStatusText` for `exporting`: show `message` verbatim when it starts with `MSG_DOWNLOADING`, else `导出中`. `resumeInterrupted()`: `uploading && uploadId != null` → `start`. `track()` DONE branch passes `onRetry` that sets `message = "$MSG_DOWNLOADING · ${reconnectingMessage(n)}"`, which `captureStatusText` shows verbatim for `exporting`. `fail()` gains `clearUploadId: Boolean = false`.

- [ ] **Step 4: Run all Android unit tests**

Run: `.\gradlew.bat :app:testDebugUnitTest`
Expected: BUILD SUCCESSFUL.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/dev/recon3d/rooms/RoomRepository.kt android/app/src/main/java/dev/recon3d/upload/UploadScreen.kt android/app/src/test/java/dev/recon3d/upload/UploadViewModelTest.kt
git commit -m "feat(app): persist upload id, resume after restart, reconnect status"
```

---

### Task 5: Docs + device test

**Files:**
- Modify: `docs/acceptance/phase1.md` (§4 item 6: new checklist lines from spec §6; §5 problem 2: resolution note with Tailscale address and date)
- Modify: `docs/setup/01-server-laptop.md` (Tailscale section: note that Windows already classes the Tailscale adapter as Private on this laptop, so the extra rule is only needed when `Get-NetConnectionProfile` shows it as Public)
- Modify: `docs/setup/03-phone.md` (address to type: `100.x.x.x` from `tailscale ip -4`)

- [ ] **Step 1: Edit the three docs** as listed.
- [ ] **Step 2: Build and install** `.\gradlew.bat :app:installDebug`; afterwards `adb reverse tcp:8765 tcp:8765` is no longer needed — set the app address to `100.81.205.31`.
- [ ] **Step 3: Restart the server** (check `/api/v1/jobs` has no active job first) and run the spec §6 device checklist; record results in phase1.md §4 item 6.
- [ ] **Step 4: Commit**

```bash
git add docs/acceptance/phase1.md docs/setup/01-server-laptop.md docs/setup/03-phone.md
git commit -m "docs: resumable transfer acceptance and Tailscale address"
```
