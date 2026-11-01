package dev.recon3d.upload

import dev.recon3d.net.Api
import dev.recon3d.net.ApiException
import dev.recon3d.net.Health
import dev.recon3d.net.JobState
import dev.recon3d.net.JobStatus
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.RoomRepository
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import kotlin.time.Duration.Companion.seconds

/** Scriptable [Api]: one-shot failures, a fixed or queued job status, fixed scene bytes. */
private class FakeApi : Api {
    var uploads = 0
    var polls = 0
    private var uploadFailure: ApiException? = null
    private var jobFailure: ApiException? = null
    var jobResult: JobStatus = JobStatus("j", JobState.DONE, 1f, "完成")
    val jobQueue = ArrayDeque<JobStatus>()
    val sceneBytes = byteArrayOf(1, 2, 3, 4)

    fun failUploadOnce(e: ApiException) { uploadFailure = e }
    fun failJobOnce(e: ApiException) { jobFailure = e }

    override suspend fun health() = Health("0.1.0", "gpu", "laptop")

    override suspend fun upload(zip: File, onProgress: (Float) -> Unit): String {
        uploads++
        uploadFailure?.let { uploadFailure = null; throw it }
        onProgress(0.5f); onProgress(1f)
        return "j"
    }

    override suspend fun job(id: String): JobStatus {
        polls++
        jobFailure?.let { jobFailure = null; throw it }
        return jobQueue.removeFirstOrNull() ?: jobResult
    }

    override suspend fun downloadScene(id: String, dest: File) {
        dest.parentFile?.mkdirs()
        dest.writeBytes(sceneBytes)
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class UploadViewModelTest {
    @get:Rule val tmp = TemporaryFolder()

    private val api = FakeApi()
    private lateinit var repo: RoomRepository
    private lateinit var scenesDir: File
    private lateinit var room: String
    private val cap = "cap-20261101-120000-000"

    @Before fun setUp() {
        repo = RoomRepository(tmp.root)
        scenesDir = File(tmp.root, "rooms")
        room = repo.createRoom("卧室").id
        val zip = File(tmp.root, "captures/$room/$cap.zip").apply { parentFile!!.mkdirs(); writeBytes(byteArrayOf(9)) }
        repo.addCapture(room, CaptureRecord(id = cap, zipPath = zip.path))
    }

    private fun TestScope.vm() = UploadViewModel(api, repo, scenesDir, poll = 3.seconds, scope = this)

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
        api.failJobOnce(ApiException("连不上计算端，请检查 Wi-Fi 和地址"))
        vm.start(room, cap); advanceUntilIdle()
        val failed = repo.capture(room, cap)
        assertEquals("failed", failed.state)
        assertEquals("j", failed.jobId)

        vm.retry(room, cap); advanceUntilIdle()
        assertEquals("done", repo.capture(room, cap).state)
        assertEquals(1, api.uploads) // resumed the existing job, no re-upload
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
}
