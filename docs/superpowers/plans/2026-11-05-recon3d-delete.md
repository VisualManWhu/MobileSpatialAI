# Delete (capture / room, phone + cloud) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the phone delete a capture or a room either everywhere (server job dir + phone files + record) or on the phone only (files gone, record kept as `remote`, model re-downloadable).

**Architecture:** One new server endpoint `DELETE /api/v1/jobs/{id}` backed by `JobStore.delete`, refusing while the GPU is writing (training/exporting) or a PUT holds the upload lock. On the phone a new pure-JVM `DeleteViewModel` sequences cancel-running → server delete → local files → record, with `UploadViewModel` gaining `cancel()` and `redownload()`; Compose gets a ⋮ menu per capture row and on the room screen, and one dialog whose buttons depend on the record's state.

**Tech Stack:** FastAPI + pytest (run in WSL via `uv run`), Kotlin/Compose + OkHttp 4.12 + MockWebServer + kotlinx-coroutines-test (run with `gradlew.bat :app:testDebugUnitTest --offline`, `JAVA_HOME=C:\Program Files\Android\Android Studio\jbr`).

**Spec:** `docs/superpowers/specs/2026-11-05-recon3d-delete-design.md`

## Global Constraints

- User-visible strings Chinese, code and comments English; commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Order for cloud delete is server first, phone second; server `404` counts as deleted.
- Server refuses with `409 {"detail":"任务正在训练，完成后再删"}` for `training`/`exporting`, `409 {"detail":"该上传正在接收中"}` when the upload lock is held, `404 {"detail":"任务不存在"}` otherwise unknown; success is `204` with no body.
- Phone retries the upload-lock 409 every 1 s, at most 5 times; any other 409 or a transport error aborts with the record untouched.
- New record state `"remote"` shows as `仅云端` with button `重新下载`.
- Local file deletion failures are ignored (`File.delete()` false → continue).
- Server tests: `cd /mnt/d/projects/Recon3D/server && export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d UV_LINK_MODE=copy && uv run pytest -q -m "not gpu"` (in WSL). Phone tests: `$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'; .\gradlew.bat :app:testDebugUnitTest --offline` from `android/`.

## Review Focus

1. A DELETE racing `Worker._run`'s first `store.update(job_id, state="training")`: the job is `queued` when DELETE checks, the worker starts a millisecond later and writes status into a directory that no longer exists → `_write` raises inside the worker. Test in Task 1 (`test_delete_queued_job_is_skipped_by_worker`) covers the common order; the race itself is closed by `_run` checking `store.get` and by `delete` holding `store._lock` while it removes, so `update` (also under `_lock`) can never interleave with the rmtree.
2. The room-level cloud delete when the first capture is `uploading` and the user taps immediately after starting: cancel must join before DELETE, else the server returns the lock 409 five times and the whole room delete fails for a transient reason. Task 4 `cloudDeleteWhileUploadingCancelsThenDeletesUploadId` pins the join.
3. A `remote` record whose server job was deleted from the laptop (`curl -X DELETE`): `重新下载` → `GET /jobs/{id}` 404 → `failed "任务不存在"` with `jobId = null`; the retry button then says `重新上传` but the zip is gone → `MSG_ZIP_MISSING`. Spec accepts this; Task 3 `redownloadOfDeletedJobFailsWithJobGone` pins the message so the user is told rather than looped.
4. Deleting a capture from the room screen while the viewer for that capture is on the back stack: the viewer reads `record.scenePath!!`. Task 5 makes `RoomScreen`/`ViewerScreen` tolerate a vanished record (`return` when `repo` no longer has it) — covered by the existing `?: return` guards; verify in Task 5 Step 6 on device.
5. `rooms.json` written by an older APK without `state="remote"`: `captureStatusText` falls through to `else -> record.state` for unknown states, so nothing crashes; nothing to add.

---

### Task 1: Server `DELETE /api/v1/jobs/{id}`

**Files:**
- Modify: `server/recon3d/jobs/store.py` (new exception, `JobStore.delete`)
- Modify: `server/recon3d/jobs/worker.py:76-80` (`_run` skips deleted ids)
- Modify: `server/recon3d/api/app.py` (route; messages)
- Modify: `docs/api.md`
- Test: `server/tests/test_jobs.py`, `server/tests/test_api.py`

**Interfaces:**
- Produces: `class JobActive(Exception)` in `store.py`; `JobStore.delete(job_id: str) -> None` raising `KeyError` (unknown / bad id), `JobActive` (state training/exporting), `UploadBusy(received)` (upload lock held). Route `DELETE /api/v1/jobs/{job_id}` → `204` / `404 {"detail":"任务不存在"}` / `409 {"detail":"任务正在训练，完成后再删"}` / `409 {"detail":"该上传正在接收中","received":n}`. Constant `TRAINING = "任务正在训练，完成后再删"` in `app.py`.

- [ ] **Step 1: Write the failing store tests** (append to `server/tests/test_jobs.py`)

```python
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
    make_worker(lambda d, p: ran.append(d.name)).start()
    wait_until(lambda: store.get(j2) is not None and store.get(j2).state == "done")
    assert ran == [j2] and not store.job_dir(j1).exists()
```

Add `JobActive` to the `from recon3d.jobs.store import (...)` block.

- [ ] **Step 2: Run them**

Run (WSL): `uv run pytest tests/test_jobs.py -q -k delete`
Expected: FAIL — `ImportError: cannot import name 'JobActive'`.

- [ ] **Step 3: Implement in `store.py`**

`class JobActive(Exception): """delete() while training/exporting."""`. `JobStore.delete(self, job_id: str) -> None`: `_JOB_ID.fullmatch` else `KeyError`; `d = self.job_dir(job_id)`; `if not d.is_dir(): raise KeyError`; take `self._upload_lock(job_id)` non-blocking → `UploadBusy(self._part_size(job_id))` when held; inside it take `self._lock`; re-check `s = self.get(job_id)`; `if s is not None and s.state in ("training", "exporting"): raise JobActive(job_id)`; `shutil.rmtree(d, ignore_errors=True)`; pop `self._upload_locks[job_id]`. (`get` does not use `_lock`, so calling it under `_lock` is safe; `update` holds `_lock` for its read-modify-write, so it cannot interleave.)

- [ ] **Step 4: Make `Worker._run` skip a deleted job**

At the top of `_run`: `if store.get(job_id) is None: return` with a one-line comment (deleted while queued). Keep `queued` deletable from the store's point of view: the id stays in the `Queue`; this check drains it harmlessly.

- [ ] **Step 5: Run store tests**

Run: `uv run pytest tests/test_jobs.py -q`
Expected: all pass (previous count + 5).

- [ ] **Step 6: Write the failing API tests** (append to `server/tests/test_api.py`)

```python
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


def test_delete_resumable_upload(client):
    data = b"d" * 3000
    uid = _begin(client, data)
    _put(client, uid, 0, data[:1000])
    store = client.app.state.store
    lock = store._upload_lock(uid)
    lock.acquire()
    try:
        r = client.delete(f"/api/v1/jobs/{uid}")
        assert r.status_code == 409 and r.json() == {"detail": "该上传正在接收中", "received": 1000}
    finally:
        lock.release()
    assert client.delete(f"/api/v1/jobs/{uid}").status_code == 204
    assert client.get(f"/api/v1/uploads/{uid}").status_code == 404
    assert all(j["job_id"] != uid for j in client.get("/api/v1/jobs").json()["jobs"])
```

`client.app.state.store`: check how `test_concurrent_put_is_409` reaches the store today and use the same way (if it builds its own store, add `app.state.store = store` in `create_app` — one line — so tests can reach it).

- [ ] **Step 7: Run them**

Run: `uv run pytest tests/test_api.py -q -k delete`
Expected: FAIL with `405` (route missing).

- [ ] **Step 8: Add the route in `app.py`**

`TRAINING = "任务正在训练，完成后再删"`. `@api.delete("/jobs/{job_id}", status_code=204, response_class=Response)` → `store.delete(job_id)`; map `KeyError → HTTPException(404, NOT_FOUND)`, `JobActive → HTTPException(409, TRAINING)`, `UploadBusy as e → JSONResponse({"detail": BUSY, "received": e.received}, 409)` (import `BUSY` from `recon3d.api.resumable`). Log `log.info("deleted job %s", job_id)`. Return `Response(status_code=204)`.

- [ ] **Step 9: Run the server suite**

Run: `uv run pytest -q -m "not gpu"`
Expected: all pass (142 + 8).

- [ ] **Step 10: Document**

`docs/api.md`: new section `## DELETE /jobs/{id}` with the four responses from the Interfaces block, a note that it also deletes an in-progress resumable upload, and under the status-page paragraph: `电脑上删任务：curl -X DELETE http://localhost:8765/api/v1/jobs/<id>`.

- [ ] **Step 11: Commit**

```bash
git add server/recon3d/jobs/store.py server/recon3d/jobs/worker.py server/recon3d/api/app.py server/tests/test_jobs.py server/tests/test_api.py docs/api.md
git commit -m "feat(server): DELETE /jobs/{id} removes a job or in-progress upload"
```

---

### Task 2: `Api.deleteJob` + `RoomRepository.removeRoom`

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/net/ApiClient.kt` (interface + impl)
- Modify: `android/app/src/main/java/dev/recon3d/AppGraph.kt:44-54` (`SettingsBackedApi`)
- Modify: `android/app/src/main/java/dev/recon3d/rooms/RoomRepository.kt`
- Test: `android/app/src/test/java/dev/recon3d/net/ApiClientTest.kt`, `android/app/src/test/java/dev/recon3d/rooms/RoomRepositoryTest.kt`, `android/app/src/test/java/dev/recon3d/upload/UploadViewModelTest.kt` (FakeApi must implement the new method to compile)

**Interfaces:**
- Produces: `suspend fun deleteJob(jobId: String)` on `Api`: returns normally on `204` and `404`; throws `ApiException(detail, code = 409)` on 409 (detail from body; `ApiClient.MSG_BUSY` when it is the upload-lock case); other HTTP codes via existing `httpError`; transport → `ApiException(MSG_UNREACHABLE)`. `RoomRepository.removeRoom(roomId: String)` — no-op when absent, persists. `FakeApi` gains `val deletedJobs = mutableListOf<String>()` and `val deleteResponses = ArrayDeque<ApiException?>()` (null = success; an exception is thrown once per dequeued entry; empty queue = success).

- [ ] **Step 1: Failing ApiClient tests** (append in `ApiClientTest.kt`)

```kotlin
@Test fun deleteJobReturnsOn204And404() = runBlocking {
    server.enqueue(MockResponse().setResponseCode(204))
    server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"任务不存在"}"""))
    client.deleteJob("abc123abc123")
    client.deleteJob("abc123abc123")
    val req = server.takeRequest()
    assertEquals("DELETE", req.method)
    assertEquals("/api/v1/jobs/abc123abc123", req.path)
}

@Test fun deleteJob409SurfacesDetailAndCode() = runBlocking {
    server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"任务正在训练，完成后再删"}"""))
    val e = assertFailsWith<ApiException> { client.deleteJob("abc123abc123") }
    assertEquals("任务正在训练，完成后再删", e.messageZh)
    assertEquals(409, e.code)
}

@Test fun deleteJobTransportErrorIsUnreachable() = runBlocking {
    server.shutdown()
    val e = assertFailsWith<ApiException> { client.deleteJob("abc123abc123") }
    assertEquals(ApiClient.MSG_UNREACHABLE, e.messageZh)
}
```

Use whatever assertion helpers the file already imports (`assertFailsWith` from kotlin.test or the file's own pattern).

- [ ] **Step 2: Failing repository test** (append in `RoomRepositoryTest.kt`)

```kotlin
@Test fun removeRoomDropsItAndPersists() {
    val repo = RoomRepository(tmp.root)
    val a = repo.createRoom("a"); val b = repo.createRoom("b")
    repo.removeRoom(a.id)
    repo.removeRoom("room-missing")  // no-op
    assertEquals(listOf(b.id), RoomRepository(tmp.root).roomsNow().map { it.id })
}
```

- [ ] **Step 3: Run both test classes**

Run: `.\gradlew.bat :app:testDebugUnitTest --offline --tests "dev.recon3d.net.ApiClientTest" --tests "dev.recon3d.rooms.RoomRepositoryTest"`
Expected: compilation FAIL (`deleteJob` / `removeRoom` unresolved).

- [ ] **Step 4: Implement**

`Api.deleteJob`; `ApiClient.deleteJob` builds `Request.Builder().url("$base/api/v1/jobs/$jobId").delete()` and uses the existing execute helper (the one that maps `IOException → MSG_UNREACHABLE`); on response: `204`/`404` → return, else `throw httpError(response)`. `SettingsBackedApi.deleteJob` forwards. `RoomRepository.removeRoom(roomId)` = `mutate { rooms -> rooms.filterNot { it.id == roomId } }`. `FakeApi.deleteJob`: `deletedJobs += jobId; deleteResponses.removeFirstOrNull()?.let { throw it }`.

- [ ] **Step 5: Run the whole Android suite**

Run: `.\gradlew.bat :app:testDebugUnitTest --offline`
Expected: BUILD SUCCESSFUL, 145 + 4 tests.

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/java/dev/recon3d/net/ApiClient.kt android/app/src/main/java/dev/recon3d/AppGraph.kt android/app/src/main/java/dev/recon3d/rooms/RoomRepository.kt android/app/src/test
git commit -m "feat(app): Api.deleteJob and RoomRepository.removeRoom"
```

---

### Task 3: `UploadViewModel.cancel` / `redownload` and the `remote` state

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/upload/UploadScreen.kt` (`UploadViewModel`, `captureStatusText`, `retryButtonLabel` untouched, `CaptureRow` later in Task 5)
- Test: `android/app/src/test/java/dev/recon3d/upload/UploadViewModelTest.kt`

**Interfaces:**
- Consumes: `FakeApi` from Task 2.
- Produces: `suspend fun cancel(roomId: String, captureId: String)` — cancels and joins the running coroutine for that capture (no-op when none); the record is left as it is (the caller deletes it next). `fun redownload(roomId: String, captureId: String)` — `launchOnce { track(roomId, captureId, jobId) }`, requires `record.jobId != null` (else `fail(..., MSG_JOB_GONE, jobGone = true)`). `captureStatusText(state = "remote") == "仅云端"`. `resumeInterrupted` ignores `remote`. Constant `STATE_REMOTE = "remote"` in `UploadScreen.kt` (internal).

- [ ] **Step 1: Failing tests** (append in `UploadViewModelTest`)

```kotlin
@Test fun cancelStopsUploadAndJoins() = runTest {
    val vm = vm()
    api.uploadDelayMs = 10_000
    vm.start(room, cap)
    advanceTimeBy(1_000)
    assertTrue(vm.isRunning(room, cap))
    vm.cancel(room, cap)
    assertFalse(vm.isRunning(room, cap))
    assertEquals("uploading", repo.capture(room, cap).state)  // caller deletes next; cancel leaves it
    advanceUntilIdle()
    assertEquals(0, api.polls)
}

@Test fun cancelWithoutRunningCoroutineIsNoop() = runTest {
    vm().cancel(room, cap)
}

@Test fun redownloadRestoresDoneFromRemote() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "remote", jobId = "j", zipPath = "", scenePath = null) }
    val vm = vm()
    vm.redownload(room, cap); advanceUntilIdle()
    val rec = repo.capture(room, cap)
    assertEquals("done", rec.state)
    assertTrue(File(rec.scenePath!!).readBytes().contentEquals(api.sceneBytes))
    assertEquals(0, api.uploads)
}

@Test fun redownloadOfDeletedJobFailsWithJobGone() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "remote", jobId = "j", zipPath = "", scenePath = null) }
    api.failJobOnce(ApiException("任务不存在", code = 404))
    val vm = vm()
    vm.redownload(room, cap); advanceUntilIdle()
    val rec = repo.capture(room, cap)
    assertEquals("failed", rec.state)
    assertEquals("任务不存在", rec.message)
    assertNull(rec.jobId)
}

@Test fun remoteStatusTextAndResumeIgnoresIt() = runTest {
    assertEquals("仅云端", captureStatusText(CaptureRecord(id = cap, zipPath = "", state = "remote", jobId = "j")))
    repo.editCapture(room, cap) { it.copy(state = "remote", jobId = "j") }
    vm().resumeInterrupted(); advanceUntilIdle()
    assertEquals(0, api.polls)
    assertEquals("remote", repo.capture(room, cap).state)
}
```

Check `FakeApi.job()` honours `failJobOnce` with a 404 by throwing it (it does for the existing `MSG_JOB_GONE` tests); `track` already maps a non-unreachable `ApiException` to `fail(..., jobGone = messageZh == "任务不存在")` via `launchOnce`.

- [ ] **Step 2: Run** — Expected: compilation FAIL (`cancel`/`redownload` unresolved).

- [ ] **Step 3: Implement**

`cancel`: `val job = synchronized(running) { running["$roomId/$captureId"] } ?: return; job.cancelAndJoin()`. `redownload`: read `jobId`; null → `fail(roomId, captureId, MSG_JOB_GONE, jobGone = true)`; else `launchOnce(roomId, captureId) { track(roomId, captureId, jobId) }`. `captureStatusText`: `STATE_REMOTE -> "仅云端"`. `resumeInterrupted` needs no change if its `when` only matches `uploading`/`ACTIVE` — confirm and leave a comment.

- [ ] **Step 4: Run the Android suite** — Expected: 149 + 5 pass.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/dev/recon3d/upload/UploadScreen.kt android/app/src/test/java/dev/recon3d/upload/UploadViewModelTest.kt
git commit -m "feat(app): cancel/redownload in UploadViewModel, remote state"
```

---

### Task 4: `DeleteViewModel` (pure JVM)

**Files:**
- Create: `android/app/src/main/java/dev/recon3d/rooms/DeleteViewModel.kt`
- Create: `android/app/src/test/java/dev/recon3d/rooms/DeleteViewModelTest.kt`
- Modify: `android/app/src/main/java/dev/recon3d/AppGraph.kt` (hold `val deleter`)
- Modify: `android/app/src/test/java/dev/recon3d/upload/UploadViewModelTest.kt` (move `FakeApi` + `InstrumentedRepo` to `android/app/src/test/java/dev/recon3d/upload/Fakes.kt` as `internal` so `DeleteViewModelTest` can reuse them)

**Interfaces:**
- Consumes: `Api.deleteJob`, `RoomRepository.removeRoom/removeCapture/editCapture/capture/room`, `UploadViewModel.cancel/isRunning`, `HomeView.fromCaptureZip(File): HomeView?`.
- Produces:

```kotlin
enum class DeleteMode { CLOUD, LOCAL }

class DeleteViewModel(
    private val api: Api,
    private val repo: RoomRepository,
    private val uploader: UploadViewModel,
    private val capturesDir: File,   // filesDir/captures
    private val scenesDir: File,     // filesDir/rooms
    private val sleep: suspend (Long) -> Unit = { delay(it) },
) {
    /** Which dialog buttons a capture gets. */
    fun captureOptions(record: CaptureRecord): List<DeleteMode>   // see table
    fun roomOptions(room: Room): List<DeleteMode>
    suspend fun deleteCapture(roomId: String, captureId: String, mode: DeleteMode): Result<Unit>
    suspend fun deleteRoom(roomId: String, mode: DeleteMode): Result<Unit>
    companion object {
        const val MSG_ROOM_TRAINING = "有任务正在训练，完成后再删"
        const val MSG_ROOM_ACTIVE = "有任务在进行中，完成后再删"
        const val BUSY_RETRIES = 5
        const val BUSY_RETRY_MS = 1_000L
    }
}
```

`captureOptions`: never uploaded (`jobId == null && uploadId == null`) → `[LOCAL]` (dialog labels it 删除 — Task 5 decides labels from the record, not from the enum); `done && scenePath != null` → `[CLOUD, LOCAL]`; else `[CLOUD]`. `roomOptions`: empty → `[LOCAL]`; else `[CLOUD, LOCAL]`. Failures are `Result.failure(ApiException(messageZh))` so the UI reads `.messageZh`.

- [ ] **Step 1: Extract fakes**

Move `FakeApi` and `InstrumentedRepo` out of `UploadViewModelTest.kt` into `Fakes.kt` (same package `dev.recon3d.upload`, `internal class`). Run the upload tests — Expected: still pass, same count.

- [ ] **Step 2: Failing tests** (`DeleteViewModelTest.kt`; setup mirrors `UploadViewModelTest`: `TemporaryFolder`, `InstrumentedRepo(tmp.root)`, `capturesDir = File(tmp.root, "captures")`, `scenesDir = File(tmp.root, "rooms")`, one room with one capture whose zip and `.points.splat` exist; `vm()` builds `UploadViewModel(... scope = this)` and `DeleteViewModel(api, repo, uploader, capturesDir, scenesDir, sleep = { })`.)

```kotlin
@Test fun cloudDeleteCallsServerThenRemovesFilesAndRecord() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "done", jobId = "j", scenePath = scene.path) }
    assertTrue(vm().deleteCapture(room, cap, DeleteMode.CLOUD).isSuccess)
    assertEquals(listOf("j"), api.deletedJobs)
    assertFalse(zip.exists()); assertFalse(points.exists()); assertFalse(scene.exists())
    assertFalse(File(scenesDir, "$room/$cap").exists())
    assertTrue(repo.room(room).captures.isEmpty())
}

@Test fun cloudDelete404StillDeletesLocally() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "failed", jobId = "j") }
    api.deleteResponses += null  // FakeApi returns normally on 404 like the real client
    assertTrue(vm().deleteCapture(room, cap, DeleteMode.CLOUD).isSuccess)
    assertFalse(zip.exists()); assertTrue(repo.room(room).captures.isEmpty())
}

@Test fun cloudDelete409LeavesEverything() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "training", jobId = "j") }
    api.deleteResponses += ApiException("任务正在训练，完成后再删", code = 409)
    val r = vm().deleteCapture(room, cap, DeleteMode.CLOUD)
    assertEquals("任务正在训练，完成后再删", (r.exceptionOrNull() as ApiException).messageZh)
    assertTrue(zip.exists()); assertEquals(1, repo.room(room).captures.size)
}

@Test fun cloudDeleteTransportErrorLeavesEverything() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "done", jobId = "j", scenePath = scene.path) }
    api.deleteResponses += ApiException(ApiClient.MSG_UNREACHABLE)
    val r = vm().deleteCapture(room, cap, DeleteMode.CLOUD)
    assertEquals(ApiClient.MSG_UNREACHABLE, (r.exceptionOrNull() as ApiException).messageZh)
    assertTrue(zip.exists()); assertEquals("done", repo.capture(room, cap).state)
}

@Test fun cloudDeleteRetriesBusy409ThenSucceeds() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "uploading", uploadId = "u") }
    repeat(3) { api.deleteResponses += ApiException(ApiClient.MSG_BUSY, code = 409) }
    assertTrue(vm().deleteCapture(room, cap, DeleteMode.CLOUD).isSuccess)
    assertEquals(List(4) { "u" }, api.deletedJobs)
}

@Test fun cloudDeleteGivesUpAfterFiveBusy409() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "uploading", uploadId = "u") }
    repeat(6) { api.deleteResponses += ApiException(ApiClient.MSG_BUSY, code = 409) }
    val r = vm().deleteCapture(room, cap, DeleteMode.CLOUD)
    assertEquals(ApiClient.MSG_BUSY, (r.exceptionOrNull() as ApiException).messageZh)
    assertEquals(6, api.deletedJobs.size)
    assertTrue(zip.exists()); assertEquals("uploading", repo.capture(room, cap).state)
}

@Test fun cloudDeleteWhileUploadingCancelsThenDeletesUploadId() = runTest {
    api.uploadDelayMs = 60_000
    val up = uploader(); up.start(room, cap); advanceTimeBy(500)
    repo.editCapture(room, cap) { it.copy(uploadId = "u") }  // FakeApi reports onCreated("u") itself if it does; else set here
    assertTrue(up.isRunning(room, cap))
    assertTrue(vm(up).deleteCapture(room, cap, DeleteMode.CLOUD).isSuccess)
    assertFalse(up.isRunning(room, cap))
    assertEquals(listOf("u"), api.deletedJobs)
    assertTrue(repo.room(room).captures.isEmpty())
}

@Test fun neverUploadedCaptureDeletesLocallyWithoutServer() = runTest {
    assertEquals(listOf(DeleteMode.LOCAL), vm().captureOptions(repo.capture(room, cap)))
    assertTrue(vm().deleteCapture(room, cap, DeleteMode.LOCAL).isSuccess)
    assertEquals(0, api.deletedJobs.size)
    assertFalse(zip.exists()); assertTrue(repo.room(room).captures.isEmpty())
}

@Test fun localDeleteOfDoneKeepsRemoteRecordAndComputesHome() = runTest {
    // zip fixture must be a real capture zip HomeView.fromCaptureZip can read: reuse the helper HomeViewTest uses
    repo.editCapture(room, cap) { it.copy(state = "done", jobId = "j", scenePath = scene.path, home = null) }
    assertTrue(vm().deleteCapture(room, cap, DeleteMode.LOCAL).isSuccess)
    val rec = repo.capture(room, cap)
    assertEquals("remote", rec.state); assertEquals("j", rec.jobId)
    assertEquals("", rec.zipPath); assertNull(rec.pointsPath); assertNull(rec.scenePath)
    assertNotNull(rec.home)
    assertFalse(zip.exists()); assertFalse(scene.exists())
    assertEquals(0, api.deletedJobs.size)
}

@Test fun captureOptionsTable() {
    val d = vm()
    val base = repo.capture(room, cap)
    assertEquals(listOf(DeleteMode.CLOUD, DeleteMode.LOCAL), d.captureOptions(base.copy(state = "done", jobId = "j", scenePath = "s")))
    assertEquals(listOf(DeleteMode.CLOUD), d.captureOptions(base.copy(state = "uploading", uploadId = "u")))
    assertEquals(listOf(DeleteMode.CLOUD), d.captureOptions(base.copy(state = "remote", jobId = "j")))
    assertEquals(listOf(DeleteMode.CLOUD), d.captureOptions(base.copy(state = "failed", jobId = "j")))
    assertEquals(listOf(DeleteMode.LOCAL), d.roomOptions(Room("r", "x", 0)))
    assertEquals(listOf(DeleteMode.CLOUD, DeleteMode.LOCAL), d.roomOptions(repo.room(room)))
}

@Test fun roomCloudDeleteRefusesWhileTraining() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "training", jobId = "j") }
    val r = vm().deleteRoom(room, DeleteMode.CLOUD)
    assertEquals(DeleteViewModel.MSG_ROOM_TRAINING, (r.exceptionOrNull() as ApiException).messageZh)
    assertEquals(0, api.deletedJobs.size); assertEquals(1, repo.roomsNow().size)
}

@Test fun roomCloudDeleteAllowsDownloadingCapture() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "exporting", jobId = "j", message = "正在下载模型…") }
    assertTrue(vm().deleteRoom(room, DeleteMode.CLOUD).isSuccess)
    assertEquals(listOf("j"), api.deletedJobs); assertTrue(repo.roomsNow().isEmpty())
}

@Test fun roomCloudDeleteStopsAtFirstFailureKeepingRest() = runTest {
    addCapture("c2", jobId = "j2"); addCapture("c3", jobId = "j3")   // helper adds zip + record state=done
    repo.editCapture(room, cap) { it.copy(state = "done", jobId = "j1") }
    api.deleteResponses += null
    api.deleteResponses += ApiException(ApiClient.MSG_UNREACHABLE)
    val r = vm().deleteRoom(room, DeleteMode.CLOUD)
    assertTrue(r.isFailure)
    assertEquals(listOf("j1", "j2"), api.deletedJobs)
    assertEquals(listOf("c2", "c3"), repo.room(room).captures.map { it.id })
}

@Test fun roomCloudDeleteSucceedsAndRemovesDirs() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "done", jobId = "j") }
    assertTrue(vm().deleteRoom(room, DeleteMode.CLOUD).isSuccess)
    assertTrue(repo.roomsNow().isEmpty())
    assertFalse(File(capturesDir, room).exists()); assertFalse(File(scenesDir, room).exists())
}

@Test fun roomLocalDeleteNeverCallsServer() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "done", jobId = "j", scenePath = scene.path) }
    assertTrue(vm().deleteRoom(room, DeleteMode.LOCAL).isSuccess)
    assertEquals(0, api.deletedJobs.size)
    assertTrue(repo.roomsNow().isEmpty()); assertFalse(zip.exists())
}

@Test fun roomLocalDeleteRefusesWhileActive() = runTest {
    repo.editCapture(room, cap) { it.copy(state = "queued", jobId = "j") }
    val r = vm().deleteRoom(room, DeleteMode.LOCAL)
    assertEquals(DeleteViewModel.MSG_ROOM_ACTIVE, (r.exceptionOrNull() as ApiException).messageZh)
    assertEquals(1, repo.roomsNow().size)
}
```

For `localDeleteOfDoneKeepsRemoteRecordAndComputesHome` the zip must be parseable: look at `HomeViewTest.kt` for how it builds a capture zip fixture and reuse (extract a shared helper into `Fakes.kt` if needed).

- [ ] **Step 3: Run** — Expected: compilation FAIL (`DeleteViewModel` unresolved).

- [ ] **Step 4: Implement `DeleteViewModel.kt`**

`deleteCapture(CLOUD)`: `uploader.cancel(roomId, captureId)`; `val id = rec.jobId ?: rec.uploadId`; if non-null, loop `attempt in 0..BUSY_RETRIES`: `try api.deleteJob(id); break` catch `ApiException` with `code == 409 && messageZh == ApiClient.MSG_BUSY && attempt < BUSY_RETRIES` → `sleep(BUSY_RETRY_MS)`, else `return Result.failure(e)`; then `deleteLocalFiles(rec)`; `repo.removeCapture`. `deleteCapture(LOCAL)`: if never uploaded → files + `removeCapture`; else (`done`) → `home = rec.home ?: HomeView.fromCaptureZip(File(rec.zipPath))`, files, `editCapture { copy(state = STATE_REMOTE, zipPath = "", pointsPath = null, scenePath = null, progress = 0f, message = null, home = home) }`. `deleteLocalFiles(rec)`: delete `zipPath`, `pointsPath`, `scenePath`, `"$scenePath.part"`, `File(scenesDir, "$roomId/$captureId")` recursively, ignoring results. `deleteRoom(CLOUD)`: precheck `captures.any { it.state == "training" || (it.state == "exporting" && it.message?.startsWith(MSG_DOWNLOADING) != true) }` → failure `MSG_ROOM_TRAINING`; for each capture in list order `deleteCapture(CLOUD)`, return the first failure; then remove `File(capturesDir, roomId)` and `File(scenesDir, roomId)`, `repo.removeRoom`. `deleteRoom(LOCAL)`: precheck `state in ACTIVE_LOCAL = setOf("uploading","queued","training","exporting")` → `MSG_ROOM_ACTIVE`; cancel each; files for each; two dirs; `removeRoom`. Wrap all in `runCatching`-style so unexpected exceptions become `Result.failure(ApiException(MSG_UNEXPECTED))` — reuse `MSG_UNEXPECTED` from `UploadScreen.kt` (make it `internal`).

- [ ] **Step 5: Wire in `AppGraph`**

`val deleter = DeleteViewModel(api = SettingsBackedApi(settings), repo = rooms, uploader = uploader, capturesDir = File(app.filesDir, "captures"), scenesDir = File(app.filesDir, "rooms"))`.

- [ ] **Step 6: Run the Android suite** — Expected: 154 + 17 pass.

- [ ] **Step 7: Commit**

```bash
git add android/app/src/main/java/dev/recon3d/rooms/DeleteViewModel.kt android/app/src/main/java/dev/recon3d/AppGraph.kt android/app/src/main/java/dev/recon3d/upload/UploadScreen.kt android/app/src/test
git commit -m "feat(app): DeleteViewModel for capture/room delete (cloud or local)"
```

---

### Task 5: UI — ⋮ menus, dialog, 重新下载, nav wiring, docs, APK

**Files:**
- Create: `android/app/src/main/java/dev/recon3d/rooms/DeleteDialog.kt`
- Modify: `android/app/src/main/java/dev/recon3d/upload/UploadScreen.kt` (`CaptureRow`: ⋮ + 重新下载 button)
- Modify: `android/app/src/main/java/dev/recon3d/rooms/RoomListScreen.kt` (`RoomScreen` top-bar ⋮, pass delete callbacks)
- Modify: `android/app/src/main/java/dev/recon3d/nav/AppNav.kt`
- Modify: `docs/acceptance/phase1.md` §4
- Test: `android/app/src/test/java/dev/recon3d/rooms/DeleteDialogTest.kt` (pure functions only)

**Interfaces:**
- Consumes: `DeleteViewModel.captureOptions/roomOptions/deleteCapture/deleteRoom`, `UploadViewModel.redownload`.
- Produces (pure, testable, in `DeleteDialog.kt`):

```kotlin
data class DeleteChoice(val mode: DeleteMode, val label: String, val destructive: Boolean)
fun captureDeleteTitle(record: CaptureRecord): String            // 删除「<id>」
fun captureDeleteBody(record: CaptureRecord, options: List<DeleteMode>): String
fun captureDeleteChoices(options: List<DeleteMode>): List<DeleteChoice>   // [CLOUD,LOCAL] → 彻底删除 / 仅删手机文件; [CLOUD] → 彻底删除; [LOCAL] → 删除
fun roomDeleteTitle(room: Room): String                          // 删除「<name>」
fun roomDeleteBody(room: Room, options: List<DeleteMode>): String
fun roomDeleteChoices(options: List<DeleteMode>): List<DeleteChoice>      // [CLOUD,LOCAL] → 彻底删除 / 仅删手机; [LOCAL] → 删除
@Composable fun DeleteDialog(title: String, body: String, choices: List<DeleteChoice>, busy: Boolean, onChoose: (DeleteMode) -> Unit, onDismiss: () -> Unit)
```

Copy strings are the spec §4.4 texts verbatim.

- [ ] **Step 1: Failing tests** (`DeleteDialogTest.kt`)

```kotlin
@Test fun choicesAndLabels() {
    assertEquals(listOf("彻底删除", "仅删手机文件"), captureDeleteChoices(listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)).map { it.label })
    assertEquals(listOf("删除"), captureDeleteChoices(listOf(DeleteMode.LOCAL)).map { it.label })
    assertEquals(listOf("彻底删除"), captureDeleteChoices(listOf(DeleteMode.CLOUD)).map { it.label })
    assertEquals(listOf("彻底删除", "仅删手机"), roomDeleteChoices(listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)).map { it.label })
    assertTrue(captureDeleteChoices(listOf(DeleteMode.CLOUD)).single().destructive)
    assertFalse(captureDeleteChoices(listOf(DeleteMode.CLOUD, DeleteMode.LOCAL))[1].destructive)
}

@Test fun bodiesFollowSpec() {
    val rec = CaptureRecord(id = "cap-1", zipPath = "")
    assertEquals("删除「cap-1」", captureDeleteTitle(rec))
    assertEquals("删除手机上的拍摄包，不可恢复。", captureDeleteBody(rec, listOf(DeleteMode.LOCAL)))
    assertEquals("手机和计算端上的拍摄包与模型都删掉，不可恢复。", captureDeleteBody(rec, listOf(DeleteMode.CLOUD)))
    assertEquals(
        "彻底删除：手机和计算端上的拍摄包与模型都删掉，不可恢复。\n仅删手机文件：释放手机空间，保留记录，之后可从计算端重新下载模型。",
        captureDeleteBody(rec, listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)),
    )
    val room = Room("r", "客厅", 0, captures = listOf(rec, rec.copy(id = "cap-2")))
    assertEquals("删除「客厅」", roomDeleteTitle(room))
    assertEquals(
        "彻底删除：这个房间的 2 条拍摄在手机和计算端都删掉，不可恢复。\n仅删手机：删掉手机上的房间和文件，计算端的任务保留。",
        roomDeleteBody(room, listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)),
    )
    assertEquals("删除手机上的这个房间，不可恢复。", roomDeleteBody(Room("r", "空", 0), listOf(DeleteMode.LOCAL)))
}
```

- [ ] **Step 2: Run** — Expected: compilation FAIL.

- [ ] **Step 3: Implement `DeleteDialog.kt`**

Pure functions per the Interfaces block; `DeleteDialog` is an `AlertDialog` with `title`, `text = body`, buttons = one `TextButton` per choice (destructive ones `colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)`), plus 取消; when `busy`, buttons disabled and a `LinearProgressIndicator` under the body.

- [ ] **Step 4: Wire the UI**

- `CaptureRow`: add params `onDelete: () -> Unit`, `onRedownload: () -> Unit`; a `Box { IconButton(⋮, contentDescription "更多") ; DropdownMenu { DropdownMenuItem("删除") } }`; `"remote" -> OutlinedButton(onRedownload) { Text("重新下载") }` in the state `when`.
- `RoomScreen`: new params `onDeleteCapture: (captureId: String) -> Unit`, `onRedownload: (captureId: String) -> Unit`, `onDeleteRoom: () -> Unit`; top-bar `actions = { IconButton(⋮, "更多") + DropdownMenu("删除房间") }`.
- `AppNav` (room route): hold `var pendingDelete by remember { mutableStateOf<DeleteTarget?>(null) }` where `sealed interface DeleteTarget { Capture(id); Room }`, `var busy`; render `DeleteDialog` when non-null using `graph.deleter.captureOptions/roomOptions`; on choose → `scope.launch { busy = true; val r = graph.deleter.deleteCapture/deleteRoom(...); busy = false; pendingDelete = null; r.exceptionOrNull()?.let { Toast(... (it as? ApiException)?.messageZh ?: MSG_UNEXPECTED) }; if room delete succeeded nav.popBackStack() }`. Cloud deletes go through `withServer { }` only when `captureOptions` contains CLOUD and the user picked it (a never-uploaded capture must be deletable with no server address saved). `onRedownload = { id -> withServer { graph.uploader.redownload(roomId, id) } }`.

- [ ] **Step 5: Build + unit tests**

Run: `.\gradlew.bat :app:testDebugUnitTest --offline` then `.\gradlew.bat :app:assembleDebug --offline`
Expected: tests pass (171 + 2), APK built.

- [ ] **Step 6: Device check (do not install while an upload is running: check `GET /api/v1/jobs` shows no `uploading` row and the app is idle)**

`adb install -r` the debug APK; on the phone: ⋮ on a done capture → 仅删手机文件 → row shows 仅云端 / 重新下载 → tap → back to 完成, model opens with the same start camera; ⋮ → 彻底删除 on a failed capture → server dir gone (`ls server/data/jobs`); room ⋮ → 删除房间 → 仅删手机 on a room with only done captures → room gone, server jobs still on the status page. Record results in `docs/acceptance/phase1.md` §4 as a new item 7 (删除).

- [ ] **Step 7: Commit**

```bash
git add android/app/src/main docs/acceptance/phase1.md android/app/src/test/java/dev/recon3d/rooms/DeleteDialogTest.kt
git commit -m "feat(app): delete capture/room from the UI (cloud or local), redownload remote models"
```
