package dev.recon3d.rooms

import dev.recon3d.viewer.HomeView
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID

/** A room the user scans; holds every capture taken in it. */
@Serializable
data class Room(
    val id: String,
    val name: String,
    /** Epoch millis. */
    val createdAt: Long,
    val captures: List<CaptureRecord> = emptyList(),
)

/** Reconstruction modes (server `mode` field, docs/api.md). */
const val MODE_FAST = "fast"
const val MODE_PRECISE = "precise"

/**
 * One capture zip and its server-side life. [state] is one of
 * `local | uploading | queued | training | exporting | done | failed`.
 */
@Serializable
data class CaptureRecord(
    val id: String,
    val zipPath: String,
    val kind: String = "model",
    /** [MODE_FAST] or [MODE_PRECISE]: the mode of the current/last server job (records from older versions: fast). */
    val mode: String = MODE_FAST,
    val jobId: String? = null,
    /** Server upload id while `uploading` (resumable protocol); kept on transport failure so retry continues. */
    val uploadId: String? = null,
    /**
     * Every server id this capture ever had (upload ids and job ids, same namespace), never cleared:
     * a failed job drops [jobId] so retry re-uploads, but its directory stays on the server until
     * "彻底删除" removes each of these.
     */
    val serverIds: List<String> = emptyList(),
    val state: String = "local",
    /** 0..1: upload fraction while `uploading`, server progress afterwards. */
    val progress: Float = 0f,
    /** Chinese status text; on failure, the reason. */
    val message: String? = null,
    /** Local `scene.splat` once downloaded. */
    val scenePath: String? = null,
    /** Preview point cloud (`.points.splat`) written at capture time; null when none. */
    val pointsPath: String? = null,
    /** Keyframes in the package (0 when unknown, e.g. records from older versions). */
    val frameCount: Int = 0,
    /** Points in [pointsPath]. */
    val pointCount: Int = 0,
    /** Viewer start camera (trajectory centre), computed from the zip on first view; null until then. */
    val home: HomeView? = null,
    /** Object recognition (phase 2): `none` | `queued` | `running` | `done` | `failed`. */
    val objectsState: String = "none",
    val objectsProgress: Float = 0f,
    /** Stage text while running; the reason when failed. */
    val objectsMessage: String? = null,
    val objectsCount: Int = 0,
    /** Directory holding the unpacked `objects.json` / `objects.bin` / `thumbs/`; kept across a failed re-run. */
    val objectsPath: String? = null,
    /** Bumped every time a new table is unpacked into [objectsPath] (the path itself never changes). */
    val objectsVersion: Int = 0,
)

/**
 * Rooms + capture records, persisted as `filesDir/rooms.json` (written to a temp file, then
 * atomically renamed over the old one). Plain JVM; all mutations are serialized on this object.
 */
open class RoomRepository(
    private val filesDir: File,
    private val clock: () -> Long = System::currentTimeMillis,
) {
    private val file = File(filesDir, "rooms.json")
    private val state = MutableStateFlow(load())

    fun rooms(): StateFlow<List<Room>> = state.asStateFlow()

    fun roomsNow(): List<Room> = state.value

    fun createRoom(name: String): Room {
        val room = Room(id = "room-" + UUID.randomUUID().toString().take(8), name = name.trim(), createdAt = clock())
        mutate { it + room }
        return room
    }

    fun addCapture(roomId: String, record: CaptureRecord) = mutateRoom(roomId) { room ->
        require(room.captures.none { it.id == record.id }) { "capture ${record.id} already in $roomId" }
        room.copy(captures = room.captures + record)
    }

    fun updateCapture(roomId: String, record: CaptureRecord) = mutateRoom(roomId) { room ->
        require(room.captures.any { it.id == record.id }) { "no capture ${record.id} in $roomId" }
        room.copy(captures = room.captures.map { if (it.id == record.id) record else it })
    }

    /** Read-modify-write of one record under the lock, so concurrent updates don't clobber each other. */
    open fun editCapture(roomId: String, captureId: String, edit: (CaptureRecord) -> CaptureRecord) = mutateRoom(roomId) { room ->
        require(room.captures.any { it.id == captureId }) { "no capture $captureId in $roomId" }
        room.copy(captures = room.captures.map { if (it.id == captureId) edit(it) else it })
    }

    /** Removes the record (the caller deletes its files). No-op if absent. */
    fun removeCapture(roomId: String, captureId: String) = mutateRoom(roomId) { room ->
        room.copy(captures = room.captures.filterNot { it.id == captureId })
    }

    /** Removes the room and its records (the caller deletes files). No-op if absent. */
    fun removeRoom(roomId: String) = mutate { rooms -> rooms.filterNot { it.id == roomId } }

    fun capture(roomId: String, captureId: String): CaptureRecord =
        room(roomId).captures.firstOrNull { it.id == captureId }
            ?: throw NoSuchElementException("no capture $captureId in $roomId")

    fun room(roomId: String): Room =
        state.value.firstOrNull { it.id == roomId } ?: throw NoSuchElementException("no room $roomId")

    private fun mutateRoom(roomId: String, change: (Room) -> Room) = mutate { rooms ->
        require(rooms.any { it.id == roomId }) { "no room $roomId" }
        rooms.map { if (it.id == roomId) change(it) else it }
    }

    @Synchronized
    private fun mutate(change: (List<Room>) -> List<Room>) {
        val next = change(state.value)
        save(next)
        state.value = next
    }

    private fun load(): List<Room> {
        if (!file.exists()) return emptyList()
        return try {
            json.decodeFromString<List<Room>>(file.readText())
        } catch (_: Exception) {
            // Keep the unreadable file for diagnosis instead of overwriting it on the next save.
            file.renameTo(File(filesDir, "rooms.json.corrupt-${clock()}"))
            emptyList()
        }
    }

    private fun save(rooms: List<Room>) {
        filesDir.mkdirs()
        val tmp = File(filesDir, "rooms.json.tmp")
        FileOutputStream(tmp).use { out ->
            out.write(json.encodeToString(rooms).toByteArray(Charsets.UTF_8))
            out.fd.sync()
        }
        Files.move(tmp.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
    }

    private companion object {
        val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    }
}
