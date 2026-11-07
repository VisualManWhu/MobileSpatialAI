package dev.recon3d.upload

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.recon3d.net.Api
import dev.recon3d.net.ApiClient
import dev.recon3d.net.ApiException
import dev.recon3d.net.JobState
import dev.recon3d.net.UploadListener
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.MODE_FAST
import dev.recon3d.rooms.MODE_PRECISE
import dev.recon3d.rooms.RoomRepository
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import kotlin.math.roundToInt
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

const val MSG_ZIP_MISSING = "拍摄文件丢失，请重新拍摄"
const val MSG_UPLOAD_INTERRUPTED = "上传被中断，请重新上传"
const val MSG_DOWNLOADING = "正在下载模型…"
const val MSG_UNEXPECTED = "出错了，请重试；如反复出现请重启应用"

/** Consecutive unreachable polls tolerated before a tracked job is marked failed (ruling R29). */
const val MAX_POLL_ERRORS = 10
private const val MSG_RETRYING_PREFIX = "网络暂时不可用，正在重试"
internal fun retryingMessage(n: Int) = "$MSG_RETRYING_PREFIX ($n/$MAX_POLL_ERRORS)"

/** Upload/download transfer is waiting to reconnect (resumable protocol); n = consecutive failures so far. */
const val MSG_RECONNECTING_PREFIX = "网络中断，正在重连"
internal fun reconnectingMessage(n: Int) = "$MSG_RECONNECTING_PREFIX ($n)"

/** Server 404 detail for an unknown job (docs/api.md): the job is gone, so retry must re-upload. */
private const val MSG_JOB_GONE = "任务不存在"
/** Status-text suffix for precise-mode records. */
const val MSG_PRECISE_SUFFIX = " · 高精"
/** Server's final message when a precise job fell back to fast reconstruction (docs/api.md). */
const val MSG_PRECISE_FALLBACK = "完成（高精位姿失败，已按快速模式重建）"
/** Prefix of the message a failed `POST /jobs/{id}/rerun` request leaves on an otherwise untouched record. */
const val MSG_RERUN_FAILED_PREFIX = "重算失败："
/** Server's plain final message; not worth keeping on the record. */
private const val MSG_DONE = "完成"
/** Server message prefix of the precise-mode pose stages (docs/api.md). */
const val MSG_POSE_STAGE_PREFIX = "位姿优化"

/** Record state after "仅删手机文件": files gone, the server still has the model (see DeleteViewModel). */
const val STATE_REMOTE = "remote"

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

    /**
     * Uploads the capture's zip in [mode] ([MODE_FAST] | [MODE_PRECISE]) and tracks the resulting job.
     * The mode is written into the record first so a retry re-uploads with the same one. No-op if it
     * is already running.
     */
    fun start(roomId: String, captureId: String, mode: String) = launchOnce(roomId, captureId) {
        repo.editCapture(roomId, captureId) { it.copy(mode = mode).withoutObjects() }
        val jobId = upload(roomId, captureId) ?: return@launchOnce
        track(roomId, captureId, jobId)
    }

    /**
     * After a failure: if the server job is still known (failure was while polling/downloading),
     * resume polling it; otherwise (upload failed, or the job itself failed/vanished) re-upload with
     * the record's mode.
     */
    fun retry(roomId: String, captureId: String) {
        val record = repo.capture(roomId, captureId)
        if (record.jobId == null) {
            start(roomId, captureId, record.mode)
        } else {
            launchOnce(roomId, captureId) { track(roomId, captureId, record.jobId) }
        }
    }

    /**
     * Re-reconstructs a finished (or failed-after-upload) capture on the server in [mode] via
     * `POST /jobs/{id}/rerun`, then tracks the new job. The local `scene.splat` stays until the new one
     * is downloaded over it; the capture is never re-uploaded. Needs a known server job, else fails
     * with [MSG_JOB_GONE].
     */
    fun rerun(roomId: String, captureId: String, mode: String) {
        val jobId = repo.capture(roomId, captureId).jobId
        if (jobId == null) {
            rerunFailed(roomId, captureId, MSG_JOB_GONE, jobGone = true)
            return
        }
        launchOnce(roomId, captureId) {
            val newId = try {
                api.rerunJob(jobId, mode)
            } catch (e: ApiException) {
                // Nothing started on the server: the capture stays as it was (done + viewable), only the
                // reason is shown. A plain 404 can also be an old server without the route, so only the
                // server's own 任务不存在 means the job is gone.
                rerunFailed(roomId, captureId, e.messageZh, jobGone = e.messageZh == MSG_JOB_GONE)
                return@launchOnce
            }
            repo.editCapture(roomId, captureId) {
                it.copy(
                    mode = mode, jobId = newId, serverIds = it.serverIds.plusDistinct(newId),
                    state = "queued", progress = 0f, message = null,
                ).withoutObjects()
            }
            track(roomId, captureId, newId)
        }
    }

    /**
     * Cancels the upload / poll / download coroutine for this capture and waits for it to end. The
     * record is left as it is: the caller (DeleteViewModel) removes it next. No-op when none runs.
     */
    suspend fun cancel(roomId: String, captureId: String) {
        val job = synchronized(running) { running["$roomId/$captureId"] } ?: return
        job.cancelAndJoin()
    }

    /** For a `remote` record: polls the known job again and downloads `scene.splat` back. */
    fun redownload(roomId: String, captureId: String) {
        val jobId = repo.capture(roomId, captureId).jobId
        if (jobId == null) {
            fail(roomId, captureId, MSG_JOB_GONE, jobGone = true)
            return
        }
        launchOnce(roomId, captureId) { track(roomId, captureId, jobId) }
    }

    /**
     * Call once at app start: uploads cut off by process death continue from the persisted upload id
     * (records from older versions without one are marked failed), and jobs that were
     * queued/training/exporting are polled again.
     */
    fun resumeInterrupted() {
        for (room in repo.roomsNow()) for (rec in room.captures) {
            when {
                rec.state == "uploading" && rec.uploadId != null -> start(room.id, rec.id, rec.mode)
                rec.state == "uploading" -> repo.editCapture(room.id, rec.id) {
                    it.copy(state = "failed", message = MSG_UPLOAD_INTERRUPTED, jobId = null)
                }
                rec.state in ACTIVE && rec.jobId != null -> retry(room.id, rec.id)
            }
        }
    }

    /**
     * Restarts the transfer a [cancel] stopped when the delete that followed it failed: an upload
     * continues from its recorded id (or from scratch), an active job is polled again. No-op otherwise.
     */
    fun resume(roomId: String, captureId: String) {
        val rec = repo.capture(roomId, captureId)
        when {
            rec.state == "uploading" -> start(roomId, captureId, rec.mode)
            rec.state in ACTIVE && rec.jobId != null -> retry(roomId, captureId)
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
                    // A server 4xx during upload means this upload can never finish; a transport
                    // failure keeps the id so "继续上传" resumes from the server's byte count.
                    val fatalUpload = e.code != null && e.code in 400..499
                    fail(roomId, captureId, e.messageZh, jobGone = e.messageZh == MSG_JOB_GONE, clearUploadId = fatalUpload)
                } catch (e: Exception) {
                    log("unexpected error for $roomId/$captureId", e)
                    fail(roomId, captureId, MSG_UNEXPECTED, jobGone = false)
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
        val resumeId = repo.capture(roomId, captureId).uploadId
        // Any upload yields a new job; a partial download of an earlier job's scene must not be appended to.
        File(scenesDir, "$roomId/$captureId/scene.splat.part").delete()
        repo.editCapture(roomId, captureId) {
            it.copy(state = "uploading", progress = if (resumeId != null) it.progress else 0f, message = null, jobId = null, scenePath = null)
        }
        // Persist only when the whole percent changes, whatever the client's callback rate.
        var lastPercent = -1
        var reconnecting = false
        // Listener callbacks run on a background thread: a throw there would escape the client, so only log.
        fun record(what: String, edit: (CaptureRecord) -> CaptureRecord) {
            try {
                repo.editCapture(roomId, captureId) { if (it.state == "uploading") edit(it) else it }
            } catch (e: Exception) {
                log("could not record $what for $roomId/$captureId", e)
            }
        }
        val mode = repo.capture(roomId, captureId).mode
        val jobId = api.upload(zip, resumeId, mode, object : UploadListener {
            override fun onCreated(uploadId: String) =
                record("upload id") { it.copy(uploadId = uploadId, progress = 0f, serverIds = it.serverIds.plusDistinct(uploadId)) }

            override fun onRetry(attempt: Int) {
                reconnecting = true
                record("reconnect") { it.copy(message = reconnectingMessage(attempt)) }
            }

            override fun onProgress(fraction: Float) {
                val percent = (fraction.coerceIn(0f, 1f) * 100).toInt()
                if (percent != lastPercent || reconnecting) {
                    lastPercent = percent
                    reconnecting = false
                    record("upload progress") { it.copy(progress = percent / 100f, message = null) }
                }
            }
        })
        repo.editCapture(roomId, captureId) {
            it.copy(state = "queued", jobId = jobId, serverIds = it.serverIds.plusDistinct(jobId), uploadId = null, progress = 0f, message = null)
        }
        return jobId
    }

    /**
     * Polls until done/failed. Connection errors ([ApiClient.MSG_UNREACHABLE]) are tolerated up to
     * [MAX_POLL_ERRORS] in a row with backoff (poll, 2×poll, … capped at [MAX_BACKOFF]), keeping the last
     * known state; the record fails only on the [MAX_POLL_ERRORS]th consecutive error.
     */
    private suspend fun track(roomId: String, captureId: String, jobId: String) {
        var errors = 0
        while (true) {
            val status = try {
                api.job(jobId)
            } catch (e: ApiException) {
                if (e.messageZh != ApiClient.MSG_UNREACHABLE) throw e
                errors++
                if (errors >= MAX_POLL_ERRORS) throw e
                repo.editCapture(roomId, captureId) {
                    if (it.state in ACTIVE) it.copy(message = retryingMessage(errors)) else it
                }
                delay(backoff(errors))
                continue
            }
            errors = 0
            when (status.state) {
                JobState.DONE -> {
                    repo.editCapture(roomId, captureId) {
                        it.copy(state = "exporting", progress = 1f, message = MSG_DOWNLOADING)
                    }
                    val dest = File(scenesDir, "$roomId/$captureId/scene.splat")
                    api.downloadScene(jobId, dest) { n ->
                        val text = if (n == 0) MSG_DOWNLOADING else "$MSG_DOWNLOADING · ${reconnectingMessage(n)}"
                        try {
                            repo.editCapture(roomId, captureId) {
                                if (it.state == "exporting") it.copy(message = text) else it
                            }
                        } catch (e: Exception) {
                            log("could not record download reconnect for $roomId/$captureId", e)
                        }
                    }
                    repo.editCapture(roomId, captureId) {
                        // Keep a non-default final note (the precise-mode fallback); plain 完成 stays null.
                        it.copy(state = "done", progress = 1f, message = status.message.takeIf { m -> m != MSG_DONE }, scenePath = dest.path)
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

    /** A rerun request that never produced a job: keep state / scene, record the reason; never throws. */
    private fun rerunFailed(roomId: String, captureId: String, reason: String, jobGone: Boolean) {
        try {
            repo.editCapture(roomId, captureId) {
                it.copy(message = MSG_RERUN_FAILED_PREFIX + reason, jobId = if (jobGone) null else it.jobId)
            }
        } catch (e: Exception) {
            log("could not record rerun failure for $roomId/$captureId: $reason", e)
        }
    }

    /** Marks the record failed; never throws (a failing disk must not crash the app from here). */
    private fun fail(roomId: String, captureId: String, message: String, jobGone: Boolean, clearUploadId: Boolean = false) {
        try {
            repo.editCapture(roomId, captureId) {
                it.copy(
                    state = "failed",
                    message = message,
                    jobId = if (jobGone) null else it.jobId,
                    uploadId = if (clearUploadId) null else it.uploadId,
                )
            }
        } catch (e: Exception) {
            log("could not record failure for $roomId/$captureId: $message", e)
        }
    }

    /** n-th consecutive error: poll × 2^(n-1), capped at [MAX_BACKOFF]. */
    private fun backoff(n: Int): Duration {
        var d = poll
        repeat(n - 1) { if (d < MAX_BACKOFF) d *= 2 }
        return minOf(d, MAX_BACKOFF)
    }

    private companion object {
        const val TAG = "UploadViewModel"
        val ACTIVE = setOf("queued", "training", "exporting")
        val MAX_BACKOFF = 30.seconds
    }
}

private fun List<String>.plusDistinct(id: String): List<String> = if (id in this) this else this + id

/** A new server job has no object table: drop the fields (the old files go with the capture dir). */
private fun CaptureRecord.withoutObjects(): CaptureRecord =
    copy(objectsState = "none", objectsProgress = 0f, objectsMessage = null, objectsCount = 0, objectsPath = null)

/** Chinese status line for a capture row, e.g. `训练中 42%` or `失败：<原因>`. */
fun captureStatusText(record: CaptureRecord): String {
    val pct = (record.progress * 100).roundToInt().coerceIn(0, 100)
    val retrying = record.message?.takeIf { it.startsWith(MSG_RETRYING_PREFIX) }
    if (retrying != null && record.state in setOf("queued", "training", "exporting")) {
        val base = captureStatusText(record.copy(message = null))
        return "$base · $retrying"
    }
    val reconnecting = record.message?.takeIf { it.startsWith(MSG_RECONNECTING_PREFIX) }
    if (reconnecting != null && record.state == "uploading") return "上传中 $pct%${modeSuffix(record)} · $reconnecting"
    return when (record.state) {
        "local" -> "本地"
        "uploading" -> "上传中 $pct%" + modeSuffix(record)
        "queued" -> "排队中" + modeSuffix(record)
        // Precise mode reports its COLMAP stages as "位姿优化 (…)" while still `training`.
        "training" -> (record.message?.takeIf { it.startsWith(MSG_POSE_STAGE_PREFIX) } ?: "训练中 $pct%") + modeSuffix(record)
        // While downloading the message is "正在下载模型…", possibly with a reconnect suffix.
        "exporting" -> record.message?.takeIf { it.startsWith(MSG_DOWNLOADING) } ?: ("导出中" + modeSuffix(record))
        // A done record carries a message only for the server's fallback note or a failed rerun request.
        "done" -> record.message ?: ("完成" + modeSuffix(record))
        STATE_REMOTE -> "仅云端"
        "failed" -> "失败：${record.message ?: "未知原因"}"
        else -> record.state
    }
}

/** ` · 高精` for precise-mode records, empty for fast ones. */
private fun modeSuffix(record: CaptureRecord): String = if (record.mode == MODE_PRECISE) MSG_PRECISE_SUFFIX else ""

/** "重试" resumes polling a known server job; "继续上传" resumes a known upload; "重新上传" sends the zip again. */
fun retryButtonLabel(record: CaptureRecord): String = when {
    record.jobId != null -> "重试"
    record.uploadId != null -> "继续上传"
    else -> "重新上传"
}

/**
 * Whether the row offers "看点云": the capture-time point cloud exists and the finished model is not there yet
 * (once done, "查看 3D" is the better view).
 */
fun showsPointsButton(record: CaptureRecord): Boolean = record.pointsPath != null && record.state != "done"

/** What a row's action button does. */
sealed interface RowAction {
    /** Upload (or re-upload) the kept zip in [mode]. */
    data class Upload(val mode: String) : RowAction
    /** Resume the known job / upload ([retryButtonLabel]). */
    object Retry : RowAction
    object View : RowAction
    object Redownload : RowAction
}

/**
 * Buttons for a row's right side, left to right, as (label, action); the last one is the primary.
 * `local` offers both modes; `failed` resumes when a server id is known, else offers both modes again.
 */
fun rowActions(record: CaptureRecord): List<Pair<String, RowAction>> = when (record.state) {
    "local" -> listOf("快速上传" to RowAction.Upload(MODE_FAST), "高精上传" to RowAction.Upload(MODE_PRECISE))
    "failed" ->
        if (record.jobId != null || record.uploadId != null) listOf(retryButtonLabel(record) to RowAction.Retry)
        else listOf("快速重传" to RowAction.Upload(MODE_FAST), "高精重传" to RowAction.Upload(MODE_PRECISE))
    "done" -> listOf("查看 3D" to RowAction.View)
    STATE_REMOTE -> listOf("重新下载" to RowAction.Redownload)
    else -> emptyList()
}

/** ⋮ menu entry that re-reconstructs a finished capture in the other mode; null when not offered. */
fun rerunMenuLabel(record: CaptureRecord): String? = when {
    record.state != "done" -> null
    effectiveMode(record) == MODE_PRECISE -> "改用快速重算"
    else -> "改用高精重算"
}

/** The mode [rerunMenuLabel] switches to: the opposite of the one the model was really built with. */
fun rerunMenuMode(record: CaptureRecord): String = if (effectiveMode(record) == MODE_PRECISE) MODE_FAST else MODE_PRECISE

/** The mode the finished model was actually built with: a precise job that fell back produced a fast one. */
private fun effectiveMode(record: CaptureRecord): String =
    if (record.state == "done" && record.message == MSG_PRECISE_FALLBACK) MODE_FAST else record.mode

/**
 * One capture: id, Chinese status, progress bar while uploading/training, its action button(s) and a ⋮ menu
 * (改用高精重算 / 改用快速重算 when done, 删除).
 */
@Composable
fun CaptureRow(
    record: CaptureRecord,
    onUpload: (mode: String) -> Unit,
    onRetry: () -> Unit,
    onView: () -> Unit,
    onViewPoints: () -> Unit,
    onDelete: () -> Unit,
    onRedownload: () -> Unit,
    onRerun: (mode: String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var menuOpen by remember { mutableStateOf(false) }
    Card(modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Column(Modifier.weight(1f)) {
                    Text(record.id, style = MaterialTheme.typography.titleSmall)
                    Text(captureStatusText(record), style = MaterialTheme.typography.bodyMedium)
                }
                Box {
                    IconButton(onClick = { menuOpen = true }, modifier = Modifier.semantics { contentDescription = "更多" }) {
                        Text("⋮", fontSize = 22.sp)
                    }
                    DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                        rerunMenuLabel(record)?.let { label ->
                            DropdownMenuItem(text = { Text(label) }, onClick = { menuOpen = false; onRerun(rerunMenuMode(record)) })
                        }
                        DropdownMenuItem(text = { Text("删除") }, onClick = { menuOpen = false; onDelete() })
                    }
                }
            }
            when (record.state) {
                "uploading", "training" -> LinearProgressIndicator(
                    progress = { record.progress.coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                )
                "queued", "exporting" -> LinearProgressIndicator(Modifier.fillMaxWidth())
            }
            // Buttons on their own line: two upload buttons plus 看点云 do not fit beside the status on a 360 dp phone.
            val actions = rowActions(record)
            val points = showsPointsButton(record)
            if (actions.isNotEmpty() || points) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    if (points) TextButton(onClick = onViewPoints) { Text("看点云") }
                    actions.forEachIndexed { i, (label, action) ->
                        val onClick: () -> Unit = {
                            when (action) {
                                is RowAction.Upload -> onUpload(action.mode)
                                RowAction.Retry -> onRetry()
                                RowAction.View -> onView()
                                RowAction.Redownload -> onRedownload()
                            }
                        }
                        if (i == actions.lastIndex) Button(onClick = onClick) { Text(label) }
                        else OutlinedButton(onClick = onClick) { Text(label) }
                    }
                }
            }
        }
    }
}
