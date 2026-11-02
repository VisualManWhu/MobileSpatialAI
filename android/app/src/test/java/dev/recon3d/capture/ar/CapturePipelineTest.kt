package dev.recon3d.capture.ar

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.util.zip.ZipFile

class CapturePipelineTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun le(v: Int) = byteArrayOf((v and 0xff).toByte(), ((v shr 8) and 0xff).toByte())

    private fun job(t: Long, depth: DepthBytes? = null) = FrameJob(
        tNs = t, width = 4, height = 2,
        y = ByteArray(8), yRowStride = 4, yPixelStride = 1,
        u = ByteArray(2), v = ByteArray(2), uvRowStride = 2, uvPixelStride = 1,
        depth = depth,
        pose = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f),
        fx = 2f, fy = 2f, cx = 2f, cy = 1f,
    )

    private fun pipeline(depth: IntArray?, encode: (FrameJob) -> ByteArray) = CapturePipeline(
        stagingDir = tmp.root.resolve("room/cap-1"), roomId = "room", captureId = "cap-1", device = "test",
        depthSize = { depth }, encodeJpeg = { job, _ -> encode(job) }, log = { _, _ -> },
    )

    @Test fun previewCloudCollectsDepthKeyframesInRange() = runBlocking {
        val p = pipeline(intArrayOf(2, 1)) { byteArrayOf(1) }
        p.submit(job(1, DepthBytes(le(1500) + le(1500), 2, 1, 4)))   // 1.5 m: both pixels valid, step 2 keeps (0,0)
        p.submit(job(2, DepthBytes(le(9000) + le(0), 2, 1, 4)))      // 9 m / invalid: nothing
        p.finish()
        assertEquals(2, p.kept.value)
        assertEquals(1, p.cloud.size)
    }

    @Test fun failedJobIsSkippedAndCaptureContinues() = runBlocking {
        val p = pipeline(null) { j -> if (j.tNs == 2L) error("encoder boom") else byteArrayOf(1, 2, 3) }
        for (t in 1L..3L) { assertTrue(p.hasRoom()); p.submit(job(t)) }
        val zip = p.finish()
        assertNotNull(zip)
        assertEquals(2, p.kept.value)
        assertEquals(1, p.failed.value)
        ZipFile(zip!!).use { z ->
            val lines = z.getInputStream(z.getEntry("frames.jsonl")).bufferedReader().readLines()
            assertEquals(2, lines.size)
            assertTrue(lines[0].contains("\"t_ns\":1")); assertTrue(lines[1].contains("\"t_ns\":3"))
            assertTrue(z.getInputStream(z.getEntry("manifest.json")).bufferedReader().readText().contains("\"frame_count\":2"))
        }
    }

    @Test fun writerUsesProvidedDepthSizeAndSkipsMismatchedDepth() = runBlocking {
        val p = pipeline(intArrayOf(2, 1)) { byteArrayOf(1) }
        p.submit(job(1, DepthBytes(le(9000) + le(0xFFFF), 2, 1, 4)))
        p.submit(job(2, DepthBytes(ByteArray(8), 4, 1, 8))) // wrong size → written without depth
        p.submit(job(3, null))
        val zip = p.finish()!!
        assertEquals(3, p.kept.value)
        assertEquals(0, p.failed.value)
        ZipFile(zip).use { z ->
            val manifest = z.getInputStream(z.getEntry("manifest.json")).bufferedReader().readText()
            assertTrue(manifest.contains("\"depth_width\":2")); assertTrue(manifest.contains("\"depth_height\":1"))
            val lines = z.getInputStream(z.getEntry("frames.jsonl")).bufferedReader().readLines()
            assertTrue(lines[0].contains("\"depth\":\"depth/000001.png\""))
            assertTrue(lines[1].contains("\"depth\":null")); assertTrue(lines[2].contains("\"depth\":null"))
        }
    }

    @Test fun depthIntrinsicsPassThroughOnlyWithDepth() = runBlocking {
        val p = pipeline(intArrayOf(2, 1)) { byteArrayOf(1) }
        val k = floatArrayOf(1.5f, 1.25f, 1f, 0.5f)
        p.submit(job(1, DepthBytes(le(9000) + le(0xFFFF), 2, 1, 4, k)))
        p.submit(job(2, DepthBytes(ByteArray(8), 4, 1, 8, k))) // wrong size → no depth, no depth intrinsics
        val zip = p.finish()!!
        ZipFile(zip).use { z ->
            val lines = z.getInputStream(z.getEntry("frames.jsonl")).bufferedReader().readLines()
            assertTrue(lines[0].contains("\"depth_fx\":1.5")); assertTrue(lines[0].contains("\"depth_cy\":0.5"))
            assertTrue(!lines[1].contains("depth_fx"))
        }
    }

    @Test fun depthIntrinsicsScaleTextureIntrinsicsToDepthSize() {
        // texture 1920x1080 -> depth 160x90 (scale 1/12)
        val k = DepthIntrinsics.scaled(floatArrayOf(1440f, 1452f), floatArrayOf(960f, 540f), intArrayOf(1920, 1080), 160, 90)
        assertEquals(120f, k[0], 1e-4f); assertEquals(121f, k[1], 1e-4f)
        assertEquals(80f, k[2], 1e-4f); assertEquals(45f, k[3], 1e-4f)
    }

    @Test fun finishWithNoFramesReturnsNull() = runBlocking {
        val p = pipeline(null) { byteArrayOf(1) }
        assertEquals(false, p.hasFrames())
        assertEquals(null, p.finish())
    }

    private fun p16(x: Float) = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, x, 0f, 0f, 1f)

    @Test fun poseHistoryLooksUpNearestTimestampWithinToleranceAndCopies() {
        val h = PoseHistory()
        val ms = 1_000_000L
        val p = p16(0.5f)
        h.record(100 * ms, p)
        p[12] = 9f // caller's buffer is reused by the renderer; the history must have copied it
        h.record(133 * ms, p16(0.7f))
        assertEquals(0.5f, h.lookup(100 * ms)!![12], 0f)
        assertEquals(0.7f, h.lookup(133 * ms)!![12], 0f)
        // Pixel 10 Pro: raw depth timestamps are ~1.5 ms before their frame's timestamp
        assertEquals(0.7f, h.lookup(133 * ms - 1_500_000L)!![12], 0f)
        assertEquals(0.5f, h.lookup(100 * ms + 7_900_000L)!![12], 0f)
        assertEquals(null, h.lookup(116 * ms)) // 16 ms from both: outside tolerance
        assertEquals(null, h.lookup(100 * ms - 8_100_000L))
        assertEquals(2, h.size)
    }

    @Test fun poseHistoryEvictsEntriesOlderThanWindow() {
        val h = PoseHistory(windowNs = 2_000_000_000L)
        val step = 33_000_000L // 30 fps
        for (i in 0 until 100) h.record(i * step, p16(i.toFloat()))
        // newest t = 99*33 ms = 3267 ms; entries older than 1267 ms are gone
        assertEquals(null, h.lookup(0))
        assertEquals(null, h.lookup(38 * step)) // 1254 ms -> 2013 ms behind: evicted
        assertEquals(39f, h.lookup(39 * step)!![12], 0f) // 1980 ms behind: kept
        assertEquals(99f, h.lookup(99 * step)!![12], 0f)
        assertTrue("about 2 s / 33 ms = 61 entries, got ${h.size}", h.size in 60..62)
    }

    @Test fun rawDepthMatcherPairsOwnPoseOnceAndRejectsStale() {
        val h = PoseHistory()
        val m = RawDepthMatcher(h)
        val ms = 1_000_000L
        h.record(100 * ms, p16(1f)); h.record(133 * ms, p16(2f)); h.record(166 * ms, p16(3f))
        assertEquals(null, m.match(50 * ms)) // never seen -> stale
        // keyframe at t=166 gets a raw image computed for t=100 (timestamped ~1.5 ms earlier): pair it
        // with the pose of t=100
        assertEquals(1f, m.match(100 * ms - 1_500_000L)!![12], 0f)
        // next keyframe still sees the same raw image -> duplicate, fall back to smooth
        assertTrue(m.isDuplicate(100 * ms - 1_500_000L))
        assertEquals(null, m.match(100 * ms - 1_500_000L))
        assertFalse(m.isDuplicate(133 * ms))
        assertEquals(2f, m.match(133 * ms)!![12], 0f)
        m.reset()
        assertFalse(m.isDuplicate(133 * ms))
        assertEquals(2f, m.match(133 * ms)!![12], 0f)
    }

    @Test fun depthPoseIsWrittenOnlyForRawDepth() = runBlocking {
        val p = pipeline(intArrayOf(2, 1)) { byteArrayOf(1) }
        val conf = byteArrayOf(10, 250.toByte())
        p.submit(job(1, DepthBytes(le(9000) + le(1200), 2, 1, 4, kind = "raw", confidence = conf, confRowStride = 2, pose = p16(0.25f))))
        p.submit(job(2, DepthBytes(le(9000) + le(1200), 2, 1, 4)))
        val zip = p.finish()!!
        ZipFile(zip).use { z ->
            val lines = z.getInputStream(z.getEntry("frames.jsonl")).bufferedReader().readLines()
            assertTrue(lines[0].contains("\"depth_pose\":[1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.25,0.0,0.0,1.0]"))
            assertFalse(lines[1].contains("depth_pose"))
        }
    }

    @Test fun rawDepthConfidenceIsStrippedAndWritten() = runBlocking {
        val p = pipeline(intArrayOf(2, 1)) { byteArrayOf(1) }
        // Y8 confidence 2x1 with rowStride 4 (padding bytes must be stripped)
        val conf = byteArrayOf(10, 250.toByte(), 0x55, 0x55)
        p.submit(job(1, DepthBytes(le(9000) + le(1200), 2, 1, 4, kind = "raw", confidence = conf, confRowStride = 4)))
        p.submit(job(2, DepthBytes(le(9000) + le(1200), 2, 1, 4)))
        val zip = p.finish()!!
        ZipFile(zip).use { z ->
            val lines = z.getInputStream(z.getEntry("frames.jsonl")).bufferedReader().readLines()
            assertTrue(lines[0].contains("\"depth_kind\":\"raw\""))
            assertTrue(lines[0].contains("\"confidence\":\"conf/000001.png\""))
            assertTrue(lines[1].contains("\"depth_kind\":\"smooth\"")); assertTrue(lines[1].contains("\"confidence\":null"))
            val img = javax.imageio.ImageIO.read(z.getInputStream(z.getEntry("conf/000001.png")))
            assertEquals(2, img.width); assertEquals(1, img.height)
            assertEquals(10, img.raster.getSample(0, 0, 0)); assertEquals(250, img.raster.getSample(1, 0, 0))
        }
    }
}
