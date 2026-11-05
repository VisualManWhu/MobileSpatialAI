package dev.recon3d.upload

import dev.recon3d.net.Api
import dev.recon3d.net.ApiException
import dev.recon3d.net.Health
import dev.recon3d.net.JobState
import dev.recon3d.net.JobStatus
import dev.recon3d.net.UploadListener
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.RoomRepository
import kotlinx.coroutines.delay
import java.io.File
import java.io.IOException

/** Scriptable [Api]: one-shot failures, a fixed or queued job status, fixed scene bytes. */
internal class FakeApi : Api {
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

    /** Ids passed to deleteJob, in order. */
    val deletedJobs = mutableListOf<String>()
    /** Per-call scripted outcome: null = success (204/404), an exception = thrown. Empty = success. */
    val deleteResponses = ArrayDeque<ApiException?>()

    override suspend fun deleteJob(jobId: String) {
        deletedJobs += jobId
        deleteResponses.removeFirstOrNull()?.let { throw it }
    }

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
internal class InstrumentedRepo(dir: File) : RoomRepository(dir) {
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
