package dev.recon3d.upload

import dev.recon3d.net.Api
import dev.recon3d.net.ApiClient
import dev.recon3d.net.ApiException
import dev.recon3d.net.Health
import dev.recon3d.net.JobState
import dev.recon3d.net.JobStatus
import dev.recon3d.net.UploadListener
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.RoomRepository
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException
import kotlin.time.Duration.Companion.seconds

/** Scriptable [Api]: one-shot failures, a fixed or queued job status, fixed scene bytes. */
private class FakeApi : Api {
    var uploads = 0
    var polls = 0
    private var uploadFailure: ApiException? = null
    private var jobFailure: ApiException? = null
    private var jobFailuresLeft = 0
    private var downloadFailure: ApiException? = null
    /** Virtual-time delay inside upload, to overlap calls. */
    var uploadDelayMs = 0L
    /** Progress values reported during upload. */
    var progressSteps: List<Float> = listOf(0.5f, 1f)
    var jobResult: JobStatus = JobStatus("j", JobState.DONE, 1f, "完成")
    val jobQueue = ArrayDeque<JobStatus>()
    val sceneBytes = byteArrayOf(1, 2, 3, 4)
    var jobThrow: Exception? = null

    fun failUploadOnce(e: ApiException) { uploadFailure = e }
    fun failJobOnce(e: ApiException) = failJobTimes(1, e)
    fun failJobTimes(n: Int, e: ApiException) { jobFailure = e; jobFailuresLeft = n }
    fun failDownloadOnce(e: ApiException) { downloadFailure = e }

    override suspend fun health() = Health("0.1.0", "gpu", "laptop")

    /** The upload id the view model passed on the last call (null = new upload). */
    var lastUploadId: String? = null
    /** Retry attempts replayed through [UploadListener.onRetry] before progress. */
    var retrySteps: List<Int> = emptyList()
    /** Retry attempts replayed through downloadScene's onRetry. */
    var downloadRetrySteps: List<Int> = emptyList()

    override suspend fun upload(zip: File, uploadId: String?, listener: UploadListener): String {
        uploads++
        lastUploadId = uploadId
        if (uploadId == null) listener.onCreated("u1")
        uploadFailure?.let { uploadFailure = null; throw it }
        retrySteps.forEach(listener::onRetry)
        if (uploadDelayMs > 0) delay(uploadDelayMs)  // suspends between the retries and the progress
        progressSteps.forEach(listener::onProgress)
        return "j"
    }

    override suspend fun job(id: String): JobStatus {
        polls++
        jobThrow?.let { jobThrow = null; throw it }
        jobFailure?.let { if (--jobFailuresLeft <= 0) jobFailure = null; throw it }
        return jobQueue.removeFirstOrNull() ?: jobResult
    }

    /** Virtual-time delay inside downloadScene after the retry steps, to observe the record mid-download. */
    var downloadDelayMs = 0L
    /** Whether a stale scene.splat.part existed when downloadScene was called. */
    var partExistedAtDownload: Boolean? = null

    override suspend fun downloadScene(id: String, dest: File, onRetry: (Int) -> Unit) {
        partExistedAtDownload = File(dest.path + ".part").exists()
        downloadRetrySteps.forEach(onRetry)
        if (downloadDelayMs > 0) delay(downloadDelayMs)
        downloadFailure?.let { downloadFailure = null; throw it }
        dest.parentFile?.mkdirs()
        dest.writeBytes(sceneBytes)
    }
}

/** Counts progress writes; can throw once on the write that marks a record failed (disk full). */
private class InstrumentedRepo(dir: File) : RoomRepository(dir) {
    var throwOnFailedWrite = false
    var uploadingWrites = 0

    override fun editCapture(roomId: String, captureId: String, edit: (CaptureRecord) -> CaptureRecord) {
        val next = edit(capture(roomId, captureId))
        if (next.state == "failed" && throwOnFailedWrite) {
            throwOnFailedWrite = false
            throw IOException("No space left on device")
        }
        if (next.state == "uploading") uploadingWrites++
        super.editCapture(roomId, captureId, edit)
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class UploadViewModelTest {
    @get:Rule val tmp = TemporaryFolder()

    private val api = FakeApi()
    private lateinit var repo: InstrumentedRepo
    private lateinit var scenesDir: File
    private lateinit var room: String
    private val cap = "cap-20261101-120000-000"

    private val logged = mutableListOf<Throwable>()

    @Before fun setUp() {
        repo = InstrumentedRepo(tmp.root)
        scenesDir = File(tmp.root, "rooms")
        room = repo.createRoom("卧室").id
        val zip = File(tmp.root, "captures/$room/$cap.zip").apply { parentFile!!.mkdirs(); writeBytes(byteArrayOf(9)) }
        repo.addCapture(room, CaptureRecord(id = cap, zipPath = zip.path))
    }

    private fun TestScope.vm() =
        UploadViewModel(api, repo, scenesDir, poll = 3.seconds, log = { _, e -> logged += e }, scope = this)

    @Test fun uploadFailureKeepsZipAndAllowsRetry() = runTest {
        val vm = vm()
        api.failUploadOnce(ApiException("连不上计算端，请检查 Wi-Fi 和地址"))
        vm.start(room, cap); advanceUntilIdle()
        assertEquals("failed", repo.capture(room, cap).state)
        assertEquals("连不上计算端，请检查 Wi-Fi 和地址", repo.capture(room, cap).message)
        assertTrue(File(repo.capture(room, cap).zipPath).exists())
        vm.retry(room, cap); advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
        assertEquals(2, api.uploads)
    }

    @Test fun serverFailedJobShowsMessage() = runTest {
        val vm = vm()
        api.jobResult = JobStatus("j", JobState.FAILED, 0f, "显存不足，请降低分辨率或改用服务器")
        vm.start(room, cap); advanceUntilIdle()
        assertEquals("failed", repo.capture(room, cap).state)
        assertEquals("显存不足，请降低分辨率或改用服务器", repo.capture(room, cap).message)
        assertTrue(File(repo.capture(room, cap).zipPath).exists())
    }

    @Test fun serverFailedJobRetryReuploads() = runTest {
        val vm = vm()
        api.jobQueue.add(JobStatus("j", JobState.FAILED, 0f, "服务重启，任务中断，请重新上传"))
        vm.start(room, cap); advanceUntilIdle()
        vm.retry(room, cap); advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
        assertEquals(2, api.uploads)
    }

    @Test fun retryResumesPollingWhenJobIdExists() = runTest {
        val vm = vm()
        api.failJobTimes(MAX_POLL_ERRORS, ApiException(ApiClient.MSG_UNREACHABLE))
        vm.start(room, cap); advanceUntilIdle()
        val failed = repo.capture(room, cap)
        assertEquals("failed", failed.state)
        assertEquals("j", failed.jobId)

        vm.retry(room, cap); advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
        assertEquals(1, api.uploads) // resumed the existing job, no re-upload
    }

    @Test fun transientPollErrorsAreRetriedWithBackoff() = runTest {
        val vm = vm()
        api.jobQueue.add(JobStatus("j", JobState.TRAINING, 0.3f, "训练中"))
        vm.start(room, cap); runCurrent()
        assertEquals("training", repo.capture(room, cap).state)
        api.failJobTimes(5, ApiException(ApiClient.MSG_UNREACHABLE))
        advanceTimeBy(3.seconds); runCurrent() // poll 2 fails -> keep last state, back off 3 s
        val rec = repo.capture(room, cap)
        assertEquals("training", rec.state)
        assertEquals(0.3f, rec.progress, 1e-6f)
        assertEquals("网络暂时不可用，正在重试 (1/10)", rec.message)
        assertEquals("训练中 30% · 网络暂时不可用，正在重试 (1/10)", captureStatusText(rec))
        advanceTimeBy(3.seconds); runCurrent() // 2nd error, next backoff 6 s
        assertEquals(3, api.polls)
        advanceTimeBy(5.seconds); runCurrent()
        assertEquals(3, api.polls) // still backing off
        advanceTimeBy(1.seconds); runCurrent()
        assertEquals(4, api.polls)
        advanceUntilIdle()
        val done = repo.capture(room, cap)
        assertEquals("done", done.state)
        assertNull(done.message)
        assertEquals(1, api.uploads)
    }

    @Test fun tenConsecutivePollErrorsFailTheCapture() = runTest {
        val vm = vm()
        api.failJobTimes(MAX_POLL_ERRORS, ApiException(ApiClient.MSG_UNREACHABLE))
        vm.start(room, cap); advanceUntilIdle()
        val rec = repo.capture(room, cap)
        assertEquals("failed", rec.state)
        assertEquals(ApiClient.MSG_UNREACHABLE, rec.message)
        assertEquals("j", rec.jobId)
        assertEquals(MAX_POLL_ERRORS, api.polls)
        assertEquals("重试", retryButtonLabel(rec))
    }

    @Test fun nineErrorsThenSuccessStillCompletes() = runTest {
        val vm = vm()
        api.failJobTimes(MAX_POLL_ERRORS - 1, ApiException(ApiClient.MSG_UNREACHABLE))
        vm.start(room, cap); advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
    }

    @Test fun unexpectedExceptionShowsFixedChineseMessage() = runTest {
        val vm = vm()
        api.jobThrow = IllegalStateException("internal detail")
        vm.start(room, cap); advanceUntilIdle()
        val rec = repo.capture(room, cap)
        assertEquals("failed", rec.state)
        assertEquals(MSG_UNEXPECTED, rec.message)
        assertTrue(logged.any { it.message == "internal detail" })
    }

    @Test fun retryButtonLabelDependsOnJobId() {
        val r = CaptureRecord(id = "c", zipPath = "z", state = "failed")
        assertEquals("重新上传", retryButtonLabel(r))
        assertEquals("重试", retryButtonLabel(r.copy(jobId = "j")))
    }

    @Test fun doneDownloadsSceneAndSetsScenePath() = runTest {
        val vm = vm()
        vm.start(room, cap); advanceUntilIdle()
        val rec = repo.capture(room, cap)
        val expected = File(scenesDir, "$room/$cap/scene.splat")
        assertEquals("done", rec.state)
        assertEquals(expected.path, rec.scenePath)
        assertArrayEquals(api.sceneBytes, expected.readBytes())
    }

    @Test fun pollsEveryIntervalAndMapsProgress() = runTest {
        val vm = vm()
        api.jobQueue.add(JobStatus("j", JobState.QUEUED, 0f, "排队中"))
        api.jobQueue.add(JobStatus("j", JobState.TRAINING, 0.42f, "训练中"))
        vm.start(room, cap); runCurrent()
        assertEquals("queued", repo.capture(room, cap).state)
        assertEquals(1, api.polls)

        advanceTimeBy(3.seconds); runCurrent()
        val training = repo.capture(room, cap)
        assertEquals("training", training.state)
        assertEquals(0.42f, training.progress, 1e-6f)
        assertEquals("训练中", training.message)
        assertEquals(2, api.polls)

        advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
    }

    @Test fun unknownJobOnPollForcesReuploadOnRetry() = runTest {
        val vm = vm()
        api.failJobOnce(ApiException("任务不存在"))
        vm.start(room, cap); advanceUntilIdle()
        val rec = repo.capture(room, cap)
        assertEquals("failed", rec.state)
        assertNull(rec.jobId)
        vm.retry(room, cap); advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
        assertEquals(2, api.uploads)
    }

    @Test fun missingZipFailsWithChineseMessage() = runTest {
        val vm = vm()
        File(repo.capture(room, cap).zipPath).delete()
        vm.start(room, cap); advanceUntilIdle()
        assertEquals("failed", repo.capture(room, cap).state)
        assertEquals(MSG_ZIP_MISSING, repo.capture(room, cap).message)
        assertEquals(0, api.uploads)
    }

    @Test fun resumeInterruptedPollsActiveJobsAndFailsInterruptedUploads() = runTest {
        repo.updateCapture(room, repo.capture(room, cap).copy(state = "training", jobId = "j"))
        val other = "cap-2"
        repo.addCapture(room, CaptureRecord(id = other, zipPath = "/nope.zip", state = "uploading"))
        val vm = vm()
        vm.resumeInterrupted(); advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
        assertEquals("failed", repo.capture(room, other).state)
        assertEquals(MSG_UPLOAD_INTERRUPTED, repo.capture(room, other).message)
    }

    @Test fun failureWhileRecordingFailureIsLoggedNotThrown() = runTest {
        val vm = vm()
        api.failUploadOnce(ApiException("连不上计算端，请检查 Wi-Fi 和地址"))
        repo.throwOnFailedWrite = true
        vm.start(room, cap); advanceUntilIdle() // runTest would rethrow an escaped exception
        assertEquals(1, logged.size)
        assertTrue(logged.single() is IOException)
        assertFalse(vm.isRunning(room, cap))
        // The coroutine finished, so the capture can be started again.
        vm.start(room, cap); advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
    }

    @Test fun downloadFailureKeepsJobIdAndRetryResumesWithoutReupload() = runTest {
        val vm = vm()
        api.failDownloadOnce(ApiException("连不上计算端，请检查 Wi-Fi 和地址"))
        vm.start(room, cap); advanceUntilIdle()
        val failed = repo.capture(room, cap)
        assertEquals("failed", failed.state)
        assertNull(failed.scenePath)
        assertEquals("j", failed.jobId)

        val pollsBefore = api.polls
        vm.retry(room, cap); advanceUntilIdle()
        val done = repo.capture(room, cap)
        assertEquals(1, api.uploads)
        assertTrue(api.polls > pollsBefore)
        assertEquals("done", done.state)
        assertEquals(File(scenesDir, "$room/$cap/scene.splat").path, done.scenePath)
    }

    @Test fun concurrentStartsUploadOnce() = runTest {
        val vm = vm()
        api.uploadDelayMs = 1_000
        vm.start(room, cap); runCurrent() // first upload now suspended mid-transfer
        assertTrue(vm.isRunning(room, cap))
        vm.start(room, cap)
        vm.retry(room, cap)
        advanceUntilIdle()
        assertEquals(1, api.uploads)
        assertEquals("done", repo.capture(room, cap).state)
    }

    @Test fun uploadProgressPersistedOnlyOnWholePercentChange() = runTest {
        val vm = vm()
        api.progressSteps = (1..1000).map { it / 1000f } // 1000 callbacks, 100 distinct percents
        vm.start(room, cap); advanceUntilIdle()
        // 1 write entering "uploading", 1 for the upload id, then one per percent 0..100.
        assertEquals(103, repo.uploadingWrites)
        assertEquals("done", repo.capture(room, cap).state)
    }

    // --- resumable upload (spec 2026-11-05-recon3d-resumable-transfer-design.md §4) ---

    @Test fun uploadIdIsPersistedThenClearedOnJobCreation() = runTest {
        val vm = vm()
        api.progressSteps = listOf(0.3f); api.uploadDelayMs = 100
        vm.start(room, cap); advanceTimeBy(50); runCurrent()
        assertEquals("u1", repo.capture(room, cap).uploadId)
        advanceUntilIdle()
        val rec = repo.capture(room, cap)
        assertNull(rec.uploadId); assertEquals("j", rec.jobId); assertEquals("done", rec.state)
    }

    @Test fun resumeInterruptedContinuesUploadWithStoredId() = runTest {
        repo.editCapture(room, cap) { it.copy(state = "uploading", uploadId = "u9", progress = 0.4f) }
        vm().resumeInterrupted(); advanceUntilIdle()
        assertEquals("u9", api.lastUploadId)
        assertEquals("done", repo.capture(room, cap).state)
    }

    @Test fun resumeInterruptedWithoutIdStillFails() = runTest {  // records written by older app versions
        repo.editCapture(room, cap) { it.copy(state = "uploading") }
        vm().resumeInterrupted(); advanceUntilIdle()
        val rec = repo.capture(room, cap)
        assertEquals("failed", rec.state); assertEquals(MSG_UPLOAD_INTERRUPTED, rec.message)
        assertEquals(0, api.uploads)
    }

    @Test fun reconnectMessageShownDuringUploadAndClearedByProgress() = runTest {
        api.retrySteps = listOf(1, 2); api.uploadDelayMs = 100; api.progressSteps = listOf(0.37f)
        val vm = vm()
        vm.start(room, cap); advanceTimeBy(50); runCurrent()
        val rec = repo.capture(room, cap)
        assertEquals("uploading", rec.state)
        assertEquals(reconnectingMessage(2), rec.message)
        assertEquals("上传中 0% · 网络中断，正在重连 (2)", captureStatusText(rec))
        advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
        assertNull(repo.capture(room, cap).message)
    }

    @Test fun checksumFailureClearsUploadIdAndOffersReupload() = runTest {
        api.failUploadOnce(ApiException("文件校验失败，请重新上传", code = 422))
        val vm = vm(); vm.start(room, cap); advanceUntilIdle()
        val rec = repo.capture(room, cap)
        assertEquals("failed", rec.state); assertEquals("文件校验失败，请重新上传", rec.message)
        assertNull(rec.uploadId); assertEquals("重新上传", retryButtonLabel(rec))
    }

    @Test fun transportFailureKeepsUploadIdAndOffersContinue() = runTest {
        api.failUploadOnce(ApiException(ApiClient.MSG_UNREACHABLE))  // FakeApi calls onCreated first
        val vm = vm(); vm.start(room, cap); advanceUntilIdle()
        val rec = repo.capture(room, cap)
        assertEquals("failed", rec.state); assertEquals("u1", rec.uploadId)
        assertEquals("继续上传", retryButtonLabel(rec))
        vm.retry(room, cap); advanceUntilIdle()
        assertEquals("u1", api.lastUploadId); assertEquals("done", repo.capture(room, cap).state)
    }

    @Test fun downloadReconnectShowsMessageWithDownloading() = runTest {
        api.downloadRetrySteps = listOf(2)
        api.failDownloadOnce(ApiException(ApiClient.MSG_UNREACHABLE))
        val vm = vm(); vm.start(room, cap); advanceUntilIdle()
        val rec = repo.capture(room, cap)
        assertEquals("failed", rec.state)
        val during = rec.copy(state = "exporting", message = "$MSG_DOWNLOADING · ${reconnectingMessage(2)}")
        assertEquals("正在下载模型… · 网络中断，正在重连 (2)", captureStatusText(during))
    }

    @Test fun staleScenePartIsDroppedBeforeANewUpload() = runTest {  // review finding I1
        val part = File(scenesDir, "$room/$cap/scene.splat.part").apply { parentFile!!.mkdirs(); writeBytes(byteArrayOf(7)) }
        vm().start(room, cap); advanceUntilIdle()
        assertEquals(false, api.partExistedAtDownload)
        assertFalse(part.exists())
        assertEquals("done", repo.capture(room, cap).state)
    }

    @Test fun downloadReconnectMessageClearsWhenTransferResumes() = runTest {  // review finding I2
        api.downloadRetrySteps = listOf(2); api.downloadDelayMs = 100
        val vm = vm(); vm.start(room, cap); advanceTimeBy(50); runCurrent()
        assertEquals("正在下载模型… · 网络中断，正在重连 (2)", captureStatusText(repo.capture(room, cap)))
        api.downloadRetrySteps = emptyList()
        advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
    }

    @Test fun downloadReconnectZeroRestoresDownloadingText() = runTest {
        api.downloadRetrySteps = listOf(1, 0); api.downloadDelayMs = 100
        val vm = vm(); vm.start(room, cap); advanceTimeBy(50); runCurrent()
        assertEquals("正在下载模型…", captureStatusText(repo.capture(room, cap)))
        advanceUntilIdle()
    }

    @Test fun statusTextIsChinese() {
        val r = CaptureRecord(id = "c", zipPath = "z")
        assertEquals("本地", captureStatusText(r))
        assertEquals("上传中 42%", captureStatusText(r.copy(state = "uploading", progress = 0.42f)))
        assertEquals("排队中", captureStatusText(r.copy(state = "queued")))
        assertEquals("训练中 90%", captureStatusText(r.copy(state = "training", progress = 0.9f)))
        assertEquals("导出中", captureStatusText(r.copy(state = "exporting")))
        assertEquals("完成", captureStatusText(r.copy(state = "done")))
        assertEquals("失败：显存不足", captureStatusText(r.copy(state = "failed", message = "显存不足")))
    }

    @Test fun pointsButtonNeedsCloudAndHidesOnceDone() {
        val r = CaptureRecord(id = "c", zipPath = "z", pointsPath = "p")
        for (s in listOf("local", "uploading", "queued", "training", "exporting", "failed")) {
            assertTrue(s, showsPointsButton(r.copy(state = s)))
        }
        assertFalse(showsPointsButton(r.copy(state = "done")))
        assertFalse(showsPointsButton(r.copy(pointsPath = null, state = "training")))
    }
}
