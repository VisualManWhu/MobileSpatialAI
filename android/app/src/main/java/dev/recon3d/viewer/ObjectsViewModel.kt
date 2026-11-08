package dev.recon3d.viewer

import android.util.Log
import dev.recon3d.net.Api
import dev.recon3d.net.ApiClient
import dev.recon3d.net.ApiException
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.RoomRepository
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File
import java.util.zip.ZipFile
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

const val MSG_OBJECTS_NO_JOB = "任务已不在服务端，请先重新上传"
const val MSG_OBJECTS_BAD_ZIP = "物体表文件损坏，请重新识别"
private const val MSG_JOB_GONE = "任务不存在"
/** Server detail when recognition is already queued / running: attach to it instead of failing. */
private const val MSG_OBJECTS_ACTIVE = "正在识别"
private const val MSG_OBJECTS_UNEXPECTED = "出错了，请重试；如反复出现请重启应用"
private const val CHIP_IDLE = "识别物体"
private const val CHIP_QUEUED = "排队中"
private const val CHIP_RETRY = "重试识别"

/**
 * Phase-2 object recognition for one capture: asks the server (`POST /jobs/{id}/objects`), polls the
 * job's `objects` block, downloads `objects.zip` and unzips it next to the scene. The record's
 * `objects*` fields are the UI's source of truth; an existing table stays in place (and usable)
 * until a newer one has been fully unpacked.
 */
class ObjectsViewModel(
    private val api: Api,
    private val repo: RoomRepository,
    /** `filesDir/rooms`: `<scenesDir>/<roomId>/<captureId>/objects/` holds the unpacked table. */
    private val scenesDir: File,
    private val poll: Duration = 3.seconds,
    private val log: (String, Throwable) -> Unit = { msg, e -> Log.e(TAG, msg, e) },
    private val scope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default),
) {
    private val running = mutableMapOf<String, Job>()

    /** Requests recognition (or a re-run over an existing table) and follows it to the end. */
    fun start(roomId: String, captureId: String) = launchOnce(roomId, captureId) {
        val jobId = repo.capture(roomId, captureId).jobId
        if (jobId == null) {
            fail(roomId, captureId, MSG_OBJECTS_NO_JOB)
            return@launchOnce
        }
        val status = try {
            api.requestObjects(jobId)
        } catch (e: ApiException) {
            if (e.messageZh != MSG_OBJECTS_ACTIVE) {
                fail(roomId, captureId, e.messageZh)
                return@launchOnce
            }
            // The server is already on it (we lost track, e.g. after polling gave up): follow that run.
            null
        }
        if (status != null) {
            repo.editCapture(roomId, captureId) {
                it.copy(objectsState = status.state, objectsProgress = status.progress, objectsMessage = status.message)
            }
        }
        track(roomId, captureId, jobId)
    }

    /** Re-enters polling for a record left `queued` / `running` (screen entry, app start). */
    fun resume(roomId: String, captureId: String) {
        val rec = repo.capture(roomId, captureId)
        if (rec.objectsState !in ACTIVE) return
        val jobId = rec.jobId ?: run { fail(roomId, captureId, MSG_OBJECTS_NO_JOB); return }
        launchOnce(roomId, captureId) { track(roomId, captureId, jobId) }
    }

    private suspend fun track(roomId: String, captureId: String, jobId: String) {
        var errors = 0
        while (true) {
            val objects = try {
                api.job(jobId).objects
            } catch (e: ApiException) {
                if (e.messageZh != ApiClient.MSG_UNREACHABLE) {
                    fail(roomId, captureId, e.messageZh)
                    return
                }
                errors++
                if (errors >= MAX_POLL_ERRORS) {
                    fail(roomId, captureId, e.messageZh)
                    return
                }
                delay(backoff(errors))
                continue
            }
            errors = 0
            when (objects.state) {
                "done" -> {
                    repo.editCapture(roomId, captureId) {
                        it.copy(objectsState = "running", objectsProgress = 1f, objectsMessage = MSG_DOWNLOADING)
                    }
                    val dir = File(scenesDir, "$roomId/$captureId")
                    val zip = File(dir, "objects.zip")
                    api.downloadObjects(jobId, zip)
                    val finalDir = File(dir, "objects")
                    try {
                        unpack(zip, finalDir) // scope runs on a background dispatcher already
                    } catch (e: Exception) {
                        log("objects.zip unusable for $roomId/$captureId", e)
                        fail(roomId, captureId, MSG_OBJECTS_BAD_ZIP)
                        return
                    } finally {
                        zip.delete()
                    }
                    repo.editCapture(roomId, captureId) {
                        it.copy(
                            objectsState = "done", objectsProgress = 1f, objectsMessage = null,
                            objectsCount = objects.count, objectsPath = finalDir.path, objectsVersion = it.objectsVersion + 1,
                        )
                    }
                    return
                }
                "failed" -> {
                    fail(roomId, captureId, objects.message)
                    return
                }
                "none" -> {
                    // The server forgot the request (restarted before it was persisted): nothing to follow.
                    fail(roomId, captureId, MSG_JOB_GONE)
                    return
                }
                else -> repo.editCapture(roomId, captureId) {
                    it.copy(objectsState = objects.state, objectsProgress = objects.progress, objectsMessage = objects.message)
                }
            }
            delay(poll)
        }
    }

    /** Unzips into `<finalDir>.tmp`, checks the two required files, then atomically replaces finalDir. */
    private fun unpack(zip: File, finalDir: File) {
        val tmp = File(finalDir.path + ".tmp")
        tmp.deleteRecursively()
        tmp.mkdirs()
        val root = tmp.canonicalFile
        ZipFile(zip).use { z ->
            for (entry in z.entries()) {
                val out = File(tmp, entry.name).canonicalFile
                require(out.path.startsWith(root.path + File.separator) || out == root) { "bad entry ${entry.name}" }
                if (entry.isDirectory) {
                    out.mkdirs()
                } else {
                    out.parentFile?.mkdirs()
                    z.getInputStream(entry).use { input -> out.outputStream().use { input.copyTo(it) } }
                }
            }
        }
        require(File(tmp, ObjectsTable.JSON_NAME).isFile && File(tmp, ObjectsTable.BIN_NAME).isFile) { "incomplete objects.zip" }
        ObjectsTable.load(tmp) // throws on malformed json
        finalDir.deleteRecursively()
        if (!tmp.renameTo(finalDir)) throw java.io.IOException("rename ${tmp.path} -> ${finalDir.path} failed")
    }

    private fun fail(roomId: String, captureId: String, message: String) {
        try {
            // objectsPath is kept on purpose: an older table stays usable after a failed re-run.
            repo.editCapture(roomId, captureId) { it.copy(objectsState = "failed", objectsMessage = message) }
        } catch (e: Exception) {
            log("could not record objects failure for $roomId/$captureId", e)
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
                    fail(roomId, captureId, e.messageZh)
                } catch (e: Exception) {
                    log("unexpected error for $roomId/$captureId", e)
                    fail(roomId, captureId, MSG_OBJECTS_UNEXPECTED)
                }
            }
            running[key] = job
            job.invokeOnCompletion { synchronized(running) { if (running[key] === job) running.remove(key) } }
        }
    }

    private fun backoff(n: Int): Duration {
        var d = poll
        repeat(n - 1) { if (d < MAX_BACKOFF) d *= 2 }
        return minOf(d, MAX_BACKOFF)
    }

    private companion object {
        const val TAG = "ObjectsViewModel"
        const val MSG_DOWNLOADING = "正在下载物体表…"
        const val MAX_POLL_ERRORS = 5
        val ACTIVE = setOf("queued", "running")
        val MAX_BACKOFF = 30.seconds
    }
}

/** Text of the viewer's objects chip for this record. */
fun objectsChipLabel(record: CaptureRecord): String = when (record.objectsState) {
    "queued" -> CHIP_QUEUED
    "running" -> {
        val pct = (record.objectsProgress * 100).toInt()
        val stage = record.objectsMessage?.takeIf { it.startsWith(CHIP_IDLE) } ?: CHIP_IDLE
        "$stage $pct%"
    }
    "done" -> "物体 (${record.objectsCount})"
    "failed" -> CHIP_RETRY
    else -> CHIP_IDLE
}

/** What a tap on the objects chip does. */
enum class ChipAction { START, OPEN_LIST, NONE }

/**
 * Tap: open the list whenever a table is on disk (also during a re-run and after a failed one, so the
 * old table stays usable); otherwise start recognition unless one is already going.
 */
fun objectsChipAction(record: CaptureRecord, hasTable: Boolean): ChipAction = when {
    hasTable -> ChipAction.OPEN_LIST
    record.objectsState == "queued" || record.objectsState == "running" -> ChipAction.NONE
    else -> ChipAction.START
}

/** Long-press menu entry 「重新识别」: offered whenever a table exists and no run is in progress. */
fun objectsMenuOffersRerun(record: CaptureRecord, hasTable: Boolean): Boolean =
    hasTable && record.objectsState != "queued" && record.objectsState != "running"

/** `识别失败：<reason>` for a failed record, else null. */
fun objectsFailureText(record: CaptureRecord): String? =
    if (record.objectsState == "failed") "识别失败：${record.objectsMessage ?: ""}" else null
