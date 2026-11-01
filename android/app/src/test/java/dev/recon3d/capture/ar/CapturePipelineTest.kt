package dev.recon3d.capture.ar

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
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
        depthSize = { depth }, encodeJpeg = encode, log = { _, _ -> },
    )

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
}
