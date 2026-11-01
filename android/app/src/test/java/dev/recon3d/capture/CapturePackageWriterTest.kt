package dev.recon3d.capture

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.zip.ZipFile

class CapturePackageWriterTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun jpeg(seed: Int = 0) = ByteArray(32) { (it + seed).toByte() }
    private fun depth(w: Int, h: Int) = ShortArray(w * h) { (it * 100).toShort() }
    private fun writer(dir: File = File(tmp.root, "stage")) =
        CapturePackageWriter(dir, "room-1", "cap-1", 64, 48, 4, 3, device = "TestPhone")

    private fun entries(zip: File) = ZipFile(zip).use { z -> z.entries().asSequence().map { it.name }.toList() }
    private fun text(zip: File, name: String) =
        ZipFile(zip).use { z -> z.getInputStream(z.getEntry(name)).readBytes().decodeToString() }

    @Test fun packageMatchesFormatV1() {
        val w = writer()
        w.addFrame(1, jpeg(), depth(4, 3), identity(), 500f, 500f, 320f, 240f)
        val zip = w.finish()
        val m = Json.parseToJsonElement(text(zip, "manifest.json")).jsonObject
        assertEquals(1, m["format_version"]!!.jsonPrimitive.content.toInt())
        assertEquals(1, m["frame_count"]!!.jsonPrimitive.content.toInt())
        assertEquals("model", m["kind"]!!.jsonPrimitive.content)
        assertEquals("TestPhone", m["device"]!!.jsonPrimitive.content)
        assertEquals("room-1", m["room_id"]!!.jsonPrimitive.content)
        assertEquals("cap-1", m["capture_id"]!!.jsonPrimitive.content)
        assertEquals(64, m["image_width"]!!.jsonPrimitive.content.toInt())
        assertEquals(3, m["depth_height"]!!.jsonPrimitive.content.toInt())
        assertTrue(m["created_at"]!!.jsonPrimitive.content.matches(Regex("""\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ""")))
        assertTrue(entries(zip).containsAll(listOf("manifest.json", "frames.jsonl", "images/000001.jpg", "depth/000001.png")))
    }

    @Test fun structuralRoundTripKeepsColumnMajorPose() {
        val w = writer()
        w.addFrame(10, jpeg(1), depth(4, 3), translate(0.3f), 500f, 501f, 320f, 240f)
        w.addFrame(20, jpeg(2), null, identity(), 500f, 501f, 320f, 240f)
        val zip = w.finish()
        val lines = text(zip, "frames.jsonl").lines().filter { it.isNotBlank() }
        val m = Json.parseToJsonElement(text(zip, "manifest.json")).jsonObject
        assertEquals(lines.size, m["frame_count"]!!.jsonPrimitive.content.toInt())
        assertEquals(2, lines.size)
        val f1 = Json.parseToJsonElement(lines[0]).jsonObject
        val pose = f1["pose"]!!.jsonArray
        assertEquals(16, pose.size)
        assertEquals(0.3f, pose[12].jsonPrimitive.content.toFloat(), 0f)
        assertEquals(1, f1["index"]!!.jsonPrimitive.content.toInt())
        assertEquals(10L, f1["t_ns"]!!.jsonPrimitive.content.toLong())
        assertEquals("images/000001.jpg", f1["image"]!!.jsonPrimitive.content)
        assertEquals("depth/000001.png", f1["depth"]!!.jsonPrimitive.content)
        assertEquals(501f, f1["fy"]!!.jsonPrimitive.content.toFloat(), 0f)
        val f2 = Json.parseToJsonElement(lines[1]).jsonObject
        assertTrue(f2["depth"] is JsonNull)
        assertEquals("images/000002.jpg", f2["image"]!!.jsonPrimitive.content)
        assertFalse(entries(zip).contains("depth/000002.png"))
    }

    @Test fun finishDeletesStagingAndReturnsZip() {
        val stage = File(tmp.root, "stage")
        val w = writer(stage)
        w.addFrame(1, jpeg(), null, identity(), 1f, 1f, 1f, 1f)
        val zip = w.finish()
        assertTrue(zip.isFile)
        assertFalse(stage.exists())
    }

    /** Cross-language contract: produces the fixture consumed by server/tests/test_capture_format.py. */
    @Test fun writesSampleForServerContract() {
        // Gradle runs unit tests with cwd = android/app
        val target = listOf("../../server/tests/fixtures", "../server/tests/fixtures").map { File(it) }
            .firstOrNull { it.isDirectory } ?: return // server dir not present: skip gracefully
        val w = CapturePackageWriter(
            File(tmp.root, "sample"), "room-android", "cap-android", 64, 48, 8, 6,
            device = "Android JVM test", createdAt = "2026-11-01T08:00:00Z",
        )
        val poses = listOf(identity(), translate(0.1f), pose(10f, 0f, 0.2f, 0f, 0f))
        poses.forEachIndexed { i, p ->
            w.addFrame(
                (i + 1) * 200_000_000L, ByteArray(64) { (it * (i + 1)).toByte() },
                ShortArray(48) { (500 + it * 10 + i).toShort() }, p, 50f, 50f, 32f, 24f,
            )
        }
        val zip = w.finish()
        zip.copyTo(File(target, "android_v1.zip"), overwrite = true)
        assertEquals(3, entries(zip).count { it.startsWith("images/") })
    }
}
