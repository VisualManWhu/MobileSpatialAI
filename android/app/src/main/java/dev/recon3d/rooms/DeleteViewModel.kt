package dev.recon3d.rooms

import dev.recon3d.net.Api
import dev.recon3d.net.ApiClient
import dev.recon3d.net.ApiException
import dev.recon3d.upload.MSG_DOWNLOADING
import dev.recon3d.upload.MSG_UNEXPECTED
import dev.recon3d.upload.STATE_REMOTE
import dev.recon3d.upload.UploadViewModel
import dev.recon3d.viewer.HomeView
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import java.io.File

/** How much to delete: [CLOUD] = server job + phone; [LOCAL] = phone files only (record kept when a model exists). */
enum class DeleteMode { CLOUD, LOCAL }

/**
 * Deletes captures and rooms per docs/superpowers/specs/2026-11-05-recon3d-delete-design.md.
 * Order for a cloud delete is server first, phone second; a server 404 counts as deleted.
 * Pure JVM; every failure is `Result.failure(ApiException)` whose `messageZh` the UI shows.
 */
class DeleteViewModel(
    private val api: Api,
    private val repo: RoomRepository,
    private val uploader: UploadViewModel,
    /** `filesDir/captures`: zips and point clouds live in `<capturesDir>/<roomId>/`. */
    private val capturesDir: File,
    /** `filesDir/rooms`: downloaded models live in `<scenesDir>/<roomId>/<captureId>/`. */
    private val scenesDir: File,
    private val sleep: suspend (Long) -> Unit = { delay(it) },
) {
    /** Dialog choices for a capture, in display order. */
    fun captureOptions(record: CaptureRecord): List<DeleteMode> = when {
        record.jobId == null && record.uploadId == null -> listOf(DeleteMode.LOCAL)
        record.state == "done" && record.scenePath != null -> listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)
        else -> listOf(DeleteMode.CLOUD)
    }

    /** Dialog choices for a room, in display order. */
    fun roomOptions(room: Room): List<DeleteMode> =
        if (room.captures.isEmpty()) listOf(DeleteMode.LOCAL) else listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)

    suspend fun deleteCapture(roomId: String, captureId: String, mode: DeleteMode): Result<Unit> = guarded {
        val rec = repo.capture(roomId, captureId)
        when (mode) {
            DeleteMode.CLOUD -> {
                uploader.cancel(roomId, captureId)
                (rec.jobId ?: rec.uploadId)?.let { deleteOnServer(it) }
                deleteLocalFiles(roomId, rec)
                repo.removeCapture(roomId, captureId)
            }
            DeleteMode.LOCAL -> {
                if (rec.jobId == null && rec.uploadId == null) {
                    deleteLocalFiles(roomId, rec)
                    repo.removeCapture(roomId, captureId)
                } else {
                    // The viewer's start camera comes from the zip; compute it before the zip goes.
                    val home = rec.home ?: HomeView.fromCaptureZip(File(rec.zipPath))
                    deleteLocalFiles(roomId, rec)
                    repo.editCapture(roomId, captureId) {
                        it.copy(state = STATE_REMOTE, zipPath = "", pointsPath = null, scenePath = null, progress = 0f, message = null, home = home)
                    }
                }
            }
        }
    }

    suspend fun deleteRoom(roomId: String, mode: DeleteMode): Result<Unit> = guarded {
        val room = repo.room(roomId)
        when (mode) {
            DeleteMode.CLOUD -> {
                if (room.captures.any(::isTrainingOnServer)) throw ApiException(MSG_ROOM_TRAINING)
                for (rec in room.captures) {
                    deleteCapture(roomId, rec.id, DeleteMode.CLOUD).getOrThrow()
                }
            }
            DeleteMode.LOCAL -> {
                if (room.captures.any { it.state in ACTIVE_LOCAL }) throw ApiException(MSG_ROOM_ACTIVE)
                for (rec in room.captures) {
                    uploader.cancel(roomId, rec.id)
                    deleteLocalFiles(roomId, rec)
                }
            }
        }
        File(capturesDir, roomId).deleteRecursively()
        File(scenesDir, roomId).deleteRecursively()
        repo.removeRoom(roomId)
    }

    /** DELETE with a short retry while the server still holds the upload lock of a just-cancelled PUT. */
    private suspend fun deleteOnServer(id: String) {
        var attempt = 0
        while (true) {
            try {
                api.deleteJob(id)
                return
            } catch (e: ApiException) {
                if (e.code == 409 && e.messageZh == ApiClient.MSG_BUSY && attempt < BUSY_RETRIES) {
                    attempt++
                    sleep(BUSY_RETRY_MS)
                } else {
                    throw e
                }
            }
        }
    }

    /** Deletes the capture's files, ignoring what cannot be deleted. */
    private fun deleteLocalFiles(roomId: String, rec: CaptureRecord) {
        listOfNotNull(rec.zipPath.takeIf { it.isNotEmpty() }, rec.pointsPath, rec.scenePath, rec.scenePath?.let { "$it.part" })
            .forEach { File(it).delete() }
        File(scenesDir, "$roomId/${rec.id}").deleteRecursively()
    }

    /** `exporting` is also the phone's state while downloading a finished model; that is not training. */
    private fun isTrainingOnServer(rec: CaptureRecord): Boolean =
        rec.state == "training" || (rec.state == "exporting" && rec.message?.startsWith(MSG_DOWNLOADING) != true)

    private suspend fun guarded(block: suspend () -> Unit): Result<Unit> = try {
        block()
        Result.success(Unit)
    } catch (e: CancellationException) {
        throw e
    } catch (e: ApiException) {
        Result.failure(e)
    } catch (e: Exception) {
        Result.failure(ApiException(MSG_UNEXPECTED, e))
    }

    companion object {
        const val MSG_ROOM_TRAINING = "有任务正在训练，完成后再删"
        const val MSG_ROOM_ACTIVE = "有任务在进行中，完成后再删"
        const val BUSY_RETRIES = 5
        const val BUSY_RETRY_MS = 1_000L
        private val ACTIVE_LOCAL = setOf("uploading", "queued", "training", "exporting")
    }
}
