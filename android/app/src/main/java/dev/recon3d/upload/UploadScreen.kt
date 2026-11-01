package dev.recon3d.upload

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import dev.recon3d.net.Api
import dev.recon3d.net.ApiException
import dev.recon3d.net.JobState
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.RoomRepository
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.roundToInt
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

const val MSG_ZIP_MISSING = "拍摄文件丢失，请重新拍摄"
const val MSG_UPLOAD_INTERRUPTED = "上传被中断，请重新上传"
const val MSG_DOWNLOADING = "正在下载模型…"

/** Server 404 detail for an unknown job (docs/api.md): the job is gone, so retry must re-upload. */
private const val MSG_JOB_GONE = "任务不存在"

/**
 * Drives one capture through upload → poll every [poll] → download `scene.splat` to
 * `scenesDir/<roomId>/<captureId>/scene.splat`, writing every step into [repo].
 * Any failure sets the record to `failed` with a Chinese message; the zip is never deleted.
 * Not an androidx ViewModel: it lives in the app graph so uploads survive screen changes.
 */
class UploadViewModel(
    private val api: Api,
    private val repo: RoomRepository,
    private val scenesDir: File,
    private val poll: Duration = 3.seconds,
    /** Last-resort error log (injectable: `android.util.Log` is not available in JVM tests). */
    private val log: (String, Throwable) -> Unit = { msg, e -> Log.e(TAG, msg, e) },
    private val scope: CoroutineScope = CoroutineScope(
        SupervisorJob() + Dispatchers.Default +
            CoroutineExceptionHandler { _, e -> log("uploader coroutine crashed", e) },
    ),
) {
    private val running = mutableMapOf<String, Job>()

    /** True while an upload/poll coroutine for this capture is active. */
    fun isRunning(roomId: String, captureId: String): Boolean =
        synchronized(running) { running["$roomId/$captureId"]?.isActive == true }

    /** Uploads the capture's zip and tracks the resulting job. No-op if it is already running. */
    fun start(roomId: String, captureId: String) = launchOnce(roomId, captureId) {
        val jobId = upload(roomId, captureId) ?: return@launchOnce
        track(roomId, captureId, jobId)
    }

    /**
     * After a failure: if the server job is still known (failure was while polling/downloading),
     * resume polling it; otherwise (upload failed, or the job itself failed/vanished) re-upload.
     */
    fun retry(roomId: String, captureId: String) {
        val jobId = repo.capture(roomId, captureId).jobId
        if (jobId == null) {
            start(roomId, captureId)
        } else {
            launchOnce(roomId, captureId) { track(roomId, captureId, jobId) }
        }
    }

    /**
     * Call once at app start: uploads cut off by process death are marked failed (the stream is
     * lost), and jobs that were queued/training/exporting are polled again.
     */
    fun resumeInterrupted() {
        for (room in repo.roomsNow()) for (rec in room.captures) {
            when {
                rec.state == "uploading" -> repo.editCapture(room.id, rec.id) {
                    it.copy(state = "failed", message = MSG_UPLOAD_INTERRUPTED, jobId = null)
                }
                rec.state in ACTIVE && rec.jobId != null -> retry(room.id, rec.id)
            }
        }
    }

    private fun launchOnce(roomId: String, captureId: String, block: suspend () -> Unit) {
        val key = "$roomId/$captureId"
        synchronized(running) {
            if (running[key]?.isActive == true) return
            val job = scope.launch {
                try {
                    block()
                } catch (e: CancellationException) {
                    throw e
                } catch (e: ApiException) {
                    fail(roomId, captureId, e.messageZh, jobGone = e.messageZh == MSG_JOB_GONE)
                } catch (e: Exception) {
                    fail(roomId, captureId, "出错了：${e.message ?: e.javaClass.simpleName}", jobGone = false)
                }
            }
            running[key] = job
            job.invokeOnCompletion { synchronized(running) { if (running[key] === job) running.remove(key) } }
        }
    }

    /** Returns the job id, or null if the zip is missing (record already marked failed). */
    private suspend fun upload(roomId: String, captureId: String): String? {
        val zip = File(repo.capture(roomId, captureId).zipPath)
        if (!zip.isFile) {
            fail(roomId, captureId, MSG_ZIP_MISSING, jobGone = true)
            return null
        }
        repo.editCapture(roomId, captureId) {
            it.copy(state = "uploading", progress = 0f, message = null, jobId = null, scenePath = null)
        }
        // Persist only when the whole percent changes, whatever the client's callback rate.
        var lastPercent = 0
        val jobId = api.upload(zip) { p ->
            val percent = (p.coerceIn(0f, 1f) * 100).toInt()
            if (percent != lastPercent) {
                lastPercent = percent
                // Runs on OkHttp's writer thread: a throw here would escape OkHttp, so only log.
                try {
                    repo.editCapture(roomId, captureId) {
                        if (it.state == "uploading") it.copy(progress = percent / 100f) else it
                    }
                } catch (e: Exception) {
                    log("could not record upload progress for $roomId/$captureId", e)
                }
            }
        }
        repo.editCapture(roomId, captureId) { it.copy(state = "queued", jobId = jobId, progress = 0f, message = null) }
        return jobId
    }

    private suspend fun track(roomId: String, captureId: String, jobId: String) {
        while (true) {
            val status = api.job(jobId)
            when (status.state) {
                JobState.DONE -> {
                    repo.editCapture(roomId, captureId) {
                        it.copy(state = "exporting", progress = 1f, message = MSG_DOWNLOADING)
                    }
                    val dest = File(scenesDir, "$roomId/$captureId/scene.splat")
                    api.downloadScene(jobId, dest)
                    repo.editCapture(roomId, captureId) {
                        it.copy(state = "done", progress = 1f, message = null, scenePath = dest.path)
                    }
                    return
                }
                JobState.FAILED -> {
                    // The server job is dead; drop its id so retry re-uploads the kept zip.
                    fail(roomId, captureId, status.message, jobGone = true)
                    return
                }
                else -> repo.editCapture(roomId, captureId) {
                    it.copy(state = status.state.name.lowercase(), progress = status.progress, message = status.message)
                }
            }
            delay(poll)
        }
    }

    /** Marks the record failed; never throws (a failing disk must not crash the app from here). */
    private fun fail(roomId: String, captureId: String, message: String, jobGone: Boolean) {
        try {
            repo.editCapture(roomId, captureId) {
                it.copy(state = "failed", message = message, jobId = if (jobGone) null else it.jobId)
            }
        } catch (e: Exception) {
            log("could not record failure for $roomId/$captureId: $message", e)
        }
    }

    private companion object {
        const val TAG = "UploadViewModel"
        val ACTIVE = setOf("queued", "training", "exporting")
    }
}

/** Chinese status line for a capture row, e.g. `训练中 42%` or `失败：<原因>`. */
fun captureStatusText(record: CaptureRecord): String {
    val pct = (record.progress * 100).roundToInt().coerceIn(0, 100)
    return when (record.state) {
        "local" -> "本地"
        "uploading" -> "上传中 $pct%"
        "queued" -> "排队中"
        "training" -> "训练中 $pct%"
        "exporting" -> if (record.message == MSG_DOWNLOADING) MSG_DOWNLOADING else "导出中"
        "done" -> "完成"
        "failed" -> "失败：${record.message ?: "未知原因"}"
        else -> record.state
    }
}

/** One capture: id, Chinese status, progress bar while uploading/training, and its action button. */
@Composable
fun CaptureRow(
    record: CaptureRecord,
    onUpload: () -> Unit,
    onRetry: () -> Unit,
    onView: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Card(modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(record.id, style = MaterialTheme.typography.titleSmall)
                    Text(captureStatusText(record), style = MaterialTheme.typography.bodyMedium)
                }
                when (record.state) {
                    "local" -> Button(onClick = onUpload) { Text("上传") }
                    "failed" -> OutlinedButton(onClick = onRetry) { Text("重新上传") }
                    "done" -> Button(onClick = onView) { Text("查看 3D") }
                }
            }
            when (record.state) {
                "uploading", "training" -> LinearProgressIndicator(
                    progress = { record.progress.coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                )
                "queued", "exporting" -> LinearProgressIndicator(Modifier.fillMaxWidth())
            }
        }
    }
}
