package dev.recon3d.viewer

import dev.recon3d.net.ApiException
import dev.recon3d.net.ObjectsStatus
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.RoomRepository
import dev.recon3d.upload.FakeApi
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.time.Duration.Companion.seconds

@OptIn(ExperimentalCoroutinesApi::class)
class ObjectsViewModelTest {
    @get:Rule val tmp = TemporaryFolder()

    private val api = FakeApi()
    private lateinit var repo: RoomRepository
    private lateinit var scenesDir: File
    private lateinit var room: String
    private val cap = "cap-20261107-120000-000"

    @Before fun setUp() {
        repo = RoomRepository(tmp.root)
        scenesDir = File(tmp.root, "rooms")
        room = repo.createRoom("办公室").id
        val scene = File(scenesDir, "$room/$cap/scene.splat").apply { parentFile!!.mkdirs(); writeBytes(ByteArray(64)) }
        repo.addCapture(room, CaptureRecord(id = cap, zipPath = "", jobId = "j", state = "done", scenePath = scene.path))
        api.objectsZip = fakeZip(objectsJson(count = 1))
    }

    private fun TestScope.vm() = ObjectsViewModel(api, repo, scenesDir, poll = 3.seconds, scope = this)

    private fun rec() = repo.capture(room, cap)

    private fun objectsJson(count: Int) = """{"version":1,"job_id":"j","frames_used":2,"gaussians":2,"objects":[""" +
        (1..count).joinToString(",") {
            """{"id":$it,"label":"cup","label_zh":"杯子","score":0.7,"views":2,"center":[0,0,0],
               "bbox":{"min":[0,0,0],"max":[1,1,1]},"gaussian_count":1,"thumb":"thumbs/$it.jpg"}"""
        } + """],"relations":[]}"""

    private fun fakeZip(json: String): ByteArray {
        val out = ByteArrayOutputStream()
        ZipOutputStream(out).use { z ->
            z.putNextEntry(ZipEntry("objects.json")); z.write(json.toByteArray()); z.closeEntry()
            z.putNextEntry(ZipEntry("objects.bin")); z.write(byteArrayOf(1, 0, 0, 0)); z.closeEntry()
            z.putNextEntry(ZipEntry("thumbs/1.jpg")); z.write(byteArrayOf(9)); z.closeEntry()
        }
        return out.toByteArray()
    }

    @Test fun startRequestsPollsDownloadsAndUnzips() = runTest {
        val vm = vm()
        api.objectsStatuses.addAll(
            listOf(
                ObjectsStatus("queued", 0f, "排队中", 0),
                ObjectsStatus("running", 0.42f, "识别物体 (检测)", 0),
                ObjectsStatus("done", 1f, "完成", 1),
            ),
        )
        vm.start(room, cap); runCurrent()
        assertEquals(listOf("j"), api.requestObjectsCalls)
        assertEquals("queued", rec().objectsState)
        assertEquals("排队中", objectsChipLabel(rec()))
        advanceTimeBy(3.seconds); runCurrent()
        assertEquals("识别物体 (检测) 42%", objectsChipLabel(rec()))
        advanceUntilIdle()
        val r = rec()
        assertEquals("done", r.objectsState)
        assertEquals(1, r.objectsCount)
        assertEquals("物体 (1)", objectsChipLabel(r))
        assertTrue(r.objectsPath!!.endsWith(File.separator + "objects"))
        assertTrue(File(r.objectsPath!!, "objects.json").isFile)
        assertTrue(File(r.objectsPath!!, "objects.bin").isFile)
        assertTrue(File(r.objectsPath!!, "thumbs/1.jpg").isFile)
        assertFalse(File(scenesDir, "$room/$cap/objects.tmp").exists())
    }

    @Test fun rerunKeepsOldTableUntilNewOneLands() = runTest {
        val vm = vm()
        api.objectsStatuses.add(ObjectsStatus("done", 1f, "完成", 1))
        vm.start(room, cap); advanceUntilIdle()
        val oldPath = rec().objectsPath!!
        api.objectsZip = fakeZip(objectsJson(count = 2))
        api.objectsStatuses.addAll(listOf(ObjectsStatus("running", 0.1f, "识别物体 (检测)", 0), ObjectsStatus("done", 1f, "完成", 2)))
        vm.start(room, cap); runCurrent()
        assertEquals("running", rec().objectsState)
        assertEquals(oldPath, rec().objectsPath) // still viewable while the new run goes
        assertTrue(File(oldPath, "objects.json").readText().contains("\"id\":1"))
        advanceUntilIdle()
        assertEquals(2, rec().objectsCount)
        assertEquals(2, ObjectsTable.load(File(rec().objectsPath!!)).objects.size)
    }

    @Test fun serverFailureKeepsOldTableAndShowsReason() = runTest {
        val vm = vm()
        api.objectsStatuses.add(ObjectsStatus("done", 1f, "完成", 1))
        vm.start(room, cap); advanceUntilIdle()
        val oldPath = rec().objectsPath
        api.objectsStatuses.add(ObjectsStatus("failed", 0.3f, "显存不足，请降低分辨率或改用服务器", 0))
        vm.start(room, cap); advanceUntilIdle()
        val r = rec()
        assertEquals("failed", r.objectsState)
        assertEquals(oldPath, r.objectsPath)
        assertEquals("重试识别", objectsChipLabel(r))
        assertEquals("识别失败：显存不足，请降低分辨率或改用服务器", objectsFailureText(r))
    }

    @Test fun startWithoutServerJobFails() = runTest {
        repo.editCapture(room, cap) { it.copy(jobId = null) }
        vm().start(room, cap); advanceUntilIdle()
        assertEquals("failed", rec().objectsState)
        assertEquals(MSG_OBJECTS_NO_JOB, rec().objectsMessage)
        assertTrue(api.requestObjectsCalls.isEmpty())
    }

    @Test fun requestErrorSurfacesServerDetail() = runTest {
        api.requestObjectsFailure = ApiException("任务尚未完成", code = 409)
        vm().start(room, cap); advanceUntilIdle()
        assertEquals("failed", rec().objectsState)
        assertEquals("任务尚未完成", rec().objectsMessage)
    }

    @Test fun jobGoneWhilePollingFails() = runTest {
        api.objectsStatuses.add(ObjectsStatus("queued", 0f, "排队中", 0))
        api.failJobOnce(ApiException("任务不存在", code = 404))
        val vm = vm()
        vm.start(room, cap); runCurrent()
        advanceUntilIdle()
        assertEquals("failed", rec().objectsState)
        assertEquals("任务不存在", rec().objectsMessage)
    }

    @Test fun resumeContinuesAQueuedRecord() = runTest {
        repo.editCapture(room, cap) { it.copy(objectsState = "running", objectsProgress = 0.5f) }
        api.objectsStatuses.add(ObjectsStatus("done", 1f, "完成", 1))
        vm().resume(room, cap); advanceUntilIdle()
        assertEquals("done", rec().objectsState)
        assertTrue(api.requestObjectsCalls.isEmpty()) // resume polls, never re-requests
    }

    @Test fun chipLabels() {
        val base = CaptureRecord(id = cap, zipPath = "")
        assertEquals("识别物体", objectsChipLabel(base))
        assertEquals("排队中", objectsChipLabel(base.copy(objectsState = "queued")))
        assertEquals("识别物体 (合并) 60%", objectsChipLabel(base.copy(objectsState = "running", objectsProgress = 0.6f, objectsMessage = "识别物体 (合并)")))
        assertEquals("识别物体 7%", objectsChipLabel(base.copy(objectsState = "running", objectsProgress = 0.07f, objectsMessage = null)))
        assertEquals("物体 (0)", objectsChipLabel(base.copy(objectsState = "done", objectsCount = 0)))
        assertEquals("重试识别", objectsChipLabel(base.copy(objectsState = "failed")))
        assertNull(objectsFailureText(base))
    }
}
