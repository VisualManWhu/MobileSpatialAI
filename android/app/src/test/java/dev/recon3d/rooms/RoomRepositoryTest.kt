package dev.recon3d.rooms

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.concurrent.CountDownLatch
import kotlin.concurrent.thread

class RoomRepositoryTest {
    @get:Rule val tmp = TemporaryFolder()

    private val dir: File by lazy { tmp.root }
    private val repo by lazy { RoomRepository(dir) }

    @Test fun repositoryPersists() {
        repo.createRoom("卧室")
        assertEquals("卧室", RoomRepository(dir).roomsNow().single().name)
    }

    @Test fun capturesPersistAndUpdate() {
        val room = repo.createRoom("客厅")
        repo.addCapture(room.id, CaptureRecord(id = "cap-1", zipPath = "/x/cap-1.zip"))
        repo.updateCapture(room.id, repo.capture(room.id, "cap-1").copy(state = "queued", jobId = "j1"))

        val reloaded = RoomRepository(dir).capture(room.id, "cap-1")
        assertEquals("queued", reloaded.state)
        assertEquals("j1", reloaded.jobId)
        assertEquals("model", reloaded.kind)
    }

    @Test fun writesAtomicallyWithoutLeftoverTempFile() {
        repo.createRoom("书房")
        assertTrue(File(dir, "rooms.json").isFile)
        assertFalse(File(dir, "rooms.json.tmp").exists())
    }

    @Test fun flowEmitsCurrentRooms() {
        repo.createRoom("厨房")
        assertEquals(listOf("厨房"), repo.rooms().value.map { it.name })
    }

    @Test fun concurrentEditCaptureLosesNoUpdates() {
        val room = repo.createRoom("客厅")
        val ids = listOf("cap-a", "cap-b")
        ids.forEach { repo.addCapture(room.id, CaptureRecord(id = it, zipPath = "/x/$it.zip")) }
        val start = CountDownLatch(1)
        val threads = ids.map { id ->
            thread {
                start.await()
                repeat(50) { repo.editCapture(room.id, id) { it.copy(progress = it.progress + 1f) } }
            }
        }
        start.countDown()
        threads.forEach { it.join() }

        ids.forEach { assertEquals(50f, repo.capture(room.id, it).progress, 0f) }
        ids.forEach { assertEquals(50f, RoomRepository(dir).capture(room.id, it).progress, 0f) }
    }

    @Test fun corruptFileStartsEmptyAndIsKeptAside() {
        File(dir, "rooms.json").writeText("{not json")
        assertTrue(RoomRepository(dir).roomsNow().isEmpty())
        assertTrue(dir.listFiles()!!.any { it.name.startsWith("rooms.json.corrupt") })
    }
}
