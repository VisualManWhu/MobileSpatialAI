package dev.recon3d.rooms

import dev.recon3d.net.ApiClient
import dev.recon3d.net.ApiException
import dev.recon3d.net.JobState
import dev.recon3d.net.JobStatus
import dev.recon3d.upload.FakeApi
import dev.recon3d.upload.InstrumentedRepo
import dev.recon3d.upload.UploadViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.time.Duration.Companion.seconds

@OptIn(ExperimentalCoroutinesApi::class)
class DeleteViewModelTest {
    @get:Rule val tmp = TemporaryFolder()

    private val api = FakeApi()
    private lateinit var repo: InstrumentedRepo
    private lateinit var capturesDir: File
    private lateinit var scenesDir: File
    private lateinit var room: String
    private val cap = "cap-20261105-120000-000"
    private lateinit var zip: File
    private lateinit var points: File
    private lateinit var scene: File

    @Before fun setUp() {
        repo = InstrumentedRepo(tmp.root)
        capturesDir = File(tmp.root, "captures")
        scenesDir = File(tmp.root, "rooms")
        room = repo.createRoom("卧室").id
        zip = captureZip(cap)
        points = File(capturesDir, "$room/$cap.points.splat").apply { writeBytes(byteArrayOf(2)) }
        scene = File(scenesDir, "$room/$cap/scene.splat").apply { parentFile!!.mkdirs(); writeBytes(byteArrayOf(3)) }
        repo.addCapture(room, CaptureRecord(id = cap, zipPath = zip.path, pointsPath = points.path))
    }

    /** A minimal capture package: one identity pose in frames.jsonl, enough for HomeView.fromCaptureZip. */
    private fun captureZip(id: String): File {
        val f = File(capturesDir, "$room/$id.zip").apply { parentFile!!.mkdirs() }
        ZipOutputStream(f.outputStream()).use { z ->
            z.putNextEntry(ZipEntry("frames.jsonl"))
            val pose = (0 until 16).map { if (it % 5 == 0) 1 else 0 }.joinToString(",")
            z.write("""{"pose":[$pose]}""".toByteArray())
            z.closeEntry()
        }
        return f
    }

    private fun addCapture(id: String, jobId: String) {
        val z = captureZip(id)
        repo.addCapture(room, CaptureRecord(id = id, zipPath = z.path, state = "done", jobId = jobId))
    }

    private fun TestScope.uploader() = UploadViewModel(api, repo, scenesDir, poll = 3.seconds, log = { _, _ -> }, scope = this)

    private fun TestScope.vm(up: UploadViewModel = uploader()) =
        DeleteViewModel(api, repo, up, capturesDir, scenesDir, sleep = { })

    private fun messageOf(r: Result<Unit>) = (r.exceptionOrNull() as ApiException).messageZh

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
        api.deleteResponses += null // the real client returns normally on 404
        assertTrue(vm().deleteCapture(room, cap, DeleteMode.CLOUD).isSuccess)
        assertFalse(zip.exists()); assertTrue(repo.room(room).captures.isEmpty())
    }

    @Test fun cloudDelete409LeavesEverything() = runTest {
        repo.editCapture(room, cap) { it.copy(state = "training", jobId = "j") }
        api.deleteResponses += ApiException("任务正在训练，完成后再删", code = 409)
        val r = vm().deleteCapture(room, cap, DeleteMode.CLOUD)
        assertEquals("任务正在训练，完成后再删", messageOf(r))
        assertTrue(zip.exists()); assertEquals(1, repo.room(room).captures.size)
    }

    @Test fun cloudDeleteTransportErrorLeavesEverything() = runTest {
        repo.editCapture(room, cap) { it.copy(state = "done", jobId = "j", scenePath = scene.path) }
        api.deleteResponses += ApiException(ApiClient.MSG_UNREACHABLE)
        val r = vm().deleteCapture(room, cap, DeleteMode.CLOUD)
        assertEquals(ApiClient.MSG_UNREACHABLE, messageOf(r))
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
        assertEquals(ApiClient.MSG_BUSY, messageOf(r))
        assertEquals(6, api.deletedJobs.size)
        assertTrue(zip.exists()); assertEquals("uploading", repo.capture(room, cap).state)
    }

    @Test fun cloudDeleteWhileUploadingCancelsThenDeletesUploadId() = runTest {
        api.uploadDelayMs = 60_000
        val up = uploader()
        up.start(room, cap, "fast"); advanceTimeBy(500) // FakeApi reports onCreated("u1") before its delay
        assertTrue(up.isRunning(room, cap))
        assertEquals("u1", repo.capture(room, cap).uploadId)
        assertTrue(vm(up).deleteCapture(room, cap, DeleteMode.CLOUD).isSuccess)
        assertFalse(up.isRunning(room, cap))
        assertEquals(listOf("u1"), api.deletedJobs)
        assertTrue(repo.room(room).captures.isEmpty())
    }

    @Test fun neverUploadedCaptureDeletesLocallyWithoutServer() = runTest {
        assertEquals(listOf(DeleteMode.LOCAL), vm().captureOptions(repo.capture(room, cap)))
        assertTrue(vm().deleteCapture(room, cap, DeleteMode.LOCAL).isSuccess)
        assertEquals(0, api.deletedJobs.size)
        assertFalse(zip.exists()); assertTrue(repo.room(room).captures.isEmpty())
    }

    @Test fun localDeleteOfDoneKeepsRemoteRecordAndComputesHome() = runTest {
        repo.editCapture(room, cap) { it.copy(state = "done", jobId = "j", scenePath = scene.path, home = null) }
        assertTrue(vm().deleteCapture(room, cap, DeleteMode.LOCAL).isSuccess)
        val rec = repo.capture(room, cap)
        assertEquals("remote", rec.state); assertEquals("j", rec.jobId)
        assertEquals("", rec.zipPath); assertNull(rec.pointsPath); assertNull(rec.scenePath)
        assertNotNull(rec.home)
        assertFalse(zip.exists()); assertFalse(scene.exists())
        assertEquals(0, api.deletedJobs.size)
    }

    @Test fun captureOptionsTable() = runTest {
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
        assertEquals(DeleteViewModel.MSG_ROOM_TRAINING, messageOf(r))
        assertEquals(0, api.deletedJobs.size); assertEquals(1, repo.roomsNow().size)
    }

    @Test fun roomCloudDeleteAllowsDownloadingCapture() = runTest {
        repo.editCapture(room, cap) { it.copy(state = "exporting", jobId = "j", message = "正在下载模型…") }
        assertTrue(vm().deleteRoom(room, DeleteMode.CLOUD).isSuccess)
        assertEquals(listOf("j"), api.deletedJobs); assertTrue(repo.roomsNow().isEmpty())
    }

    @Test fun roomCloudDeleteStopsAtFirstFailureKeepingRest() = runTest {
        addCapture("c2", jobId = "j2"); addCapture("c3", jobId = "j3")
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
        assertEquals(DeleteViewModel.MSG_ROOM_ACTIVE, messageOf(r))
        assertEquals(1, repo.roomsNow().size)
    }

    // --- review fix pass ---

    @Test fun cloudDelete409OnTrainingResumesPolling() = runTest {
        repo.editCapture(room, cap) { it.copy(state = "training", jobId = "j", progress = 0.4f) }
        api.jobResult = JobStatus("j", JobState.TRAINING, 0.5f, "训练中")
        val up = uploader()
        up.retry(room, cap); advanceTimeBy(500)
        assertTrue(up.isRunning(room, cap))
        val pollsBefore = api.polls
        api.deleteResponses += ApiException("任务正在训练，完成后再删", code = 409)
        val r = vm(up).deleteCapture(room, cap, DeleteMode.CLOUD)
        assertEquals("任务正在训练，完成后再删", messageOf(r))
        assertTrue(up.isRunning(room, cap))
        advanceTimeBy(10_000)
        assertTrue(api.polls > pollsBefore)
        assertEquals("training", repo.capture(room, cap).state)
        up.cancel(room, cap)
    }

    @Test fun cloudDeleteFailureOnUploadingResumesUpload() = runTest {
        api.uploadDelayMs = 60_000
        val up = uploader()
        up.start(room, cap, "fast"); advanceTimeBy(500)
        repeat(6) { api.deleteResponses += ApiException(ApiClient.MSG_BUSY, code = 409) }
        assertTrue(vm(up).deleteCapture(room, cap, DeleteMode.CLOUD).isFailure)
        assertTrue(up.isRunning(room, cap))
        advanceTimeBy(100) // let the relaunched coroutine reach the API
        assertEquals(2, api.uploads) // the cancelled upload was started again
        assertEquals("u1", api.lastUploadId) // and continues from the recorded upload id
        up.cancel(room, cap)
    }

    @Test fun uploadingWithoutIdsYetIsCloudOnlyAndGetsCancelled() = runTest {
        api.createdLate = true
        api.uploadDelayMs = 60_000
        val up = uploader()
        up.start(room, cap, "fast"); advanceTimeBy(500)
        val rec = repo.capture(room, cap)
        assertEquals("uploading", rec.state); assertNull(rec.uploadId); assertNull(rec.jobId)
        assertEquals(listOf(DeleteMode.CLOUD), vm(up).captureOptions(rec))
        assertTrue(vm(up).deleteCapture(room, cap, DeleteMode.CLOUD).isSuccess)
        assertFalse(up.isRunning(room, cap))
        assertEquals(0, api.deletedJobs.size)
        assertFalse(zip.exists()); assertTrue(repo.room(room).captures.isEmpty())
    }

    @Test fun cloudDeleteRemovesEveryServerIdEvenWhenJobIdWasCleared() = runTest {
        repo.editCapture(room, cap) { it.copy(state = "failed", jobId = null, uploadId = null, serverIds = listOf("old1", "old2")) }
        val rec = repo.capture(room, cap)
        assertEquals(listOf(DeleteMode.CLOUD), vm().captureOptions(rec))
        assertTrue(vm().deleteCapture(room, cap, DeleteMode.CLOUD).isSuccess)
        assertEquals(listOf("old1", "old2"), api.deletedJobs)
        assertTrue(repo.room(room).captures.isEmpty())
    }
}
