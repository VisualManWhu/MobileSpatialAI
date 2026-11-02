package dev.recon3d.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PointCloudTest {
    private val dw = 4; private val dh = 2
    private val w = 8; private val h = 4
    private val k = floatArrayOf(2f, 2f, 2f, 1f) // fx, fy, cx, cy in depth pixels

    /** NV21 image where every pixel is the same colour (Y, U, V). */
    private fun nv21(y: Int, u: Int, v: Int): ByteArray {
        val b = ByteArray(w * h * 3 / 2)
        for (i in 0 until w * h) b[i] = y.toByte()
        var o = w * h
        while (o < b.size) { b[o] = v.toByte(); b[o + 1] = u.toByte(); o += 2 }
        return b
    }

    private fun depth(mm: Int) = ShortArray(dw * dh) { mm.toShort() }

    @Test fun backProjectsWithPoseAndColoursFromImage() {
        val c = DepthCloud(step = 1)
        // all pixels 2 m away; camera at x = 1 looking down -Z; grey-ish red image (Y=100, V=+40)
        val added = c.addKeyframe(depth(2000), null, dw, dh, k, translate(1f), nv21(100, 128, 168), w, h)
        assertEquals(dw * dh, added)
        val s = c.snapshot()
        // depth pixel (2, 1) is the principal point: straight ahead -> world (1, 0, -2)
        val i = 1 * dw + 2
        assertEquals(1f, s.xyz[3 * i], 1e-6f); assertEquals(0f, s.xyz[3 * i + 1], 1e-6f); assertEquals(-2f, s.xyz[3 * i + 2], 1e-6f)
        // depth pixel (3, 1): one pixel right of centre at fx = 2 -> x = +1 m in camera, world x = 2
        assertEquals(2f, s.xyz[3 * (1 * dw + 3)], 1e-6f)
        // depth pixel (2, 0): one pixel above centre (image y down) -> world y = +1
        assertEquals(1f, s.xyz[3 * (0 * dw + 2) + 1], 1e-6f)
        // colour: Y=100, V=168 -> R = 100 + 1.402*40 = 156, G = 100 - 0.714*40 = 71, B = 100
        assertEquals(156, s.rgb[0].toInt() and 0xFF); assertEquals(71, s.rgb[1].toInt() and 0xFF); assertEquals(100, s.rgb[2].toInt() and 0xFF)
    }

    @Test fun rotationOfPoseIsApplied() {
        val c = DepthCloud(step = 1)
        // camera turned 90 deg to the right: its -Z now points along world +X, so a point 2 m ahead is at world x = +2
        c.addKeyframe(depth(2000), null, dw, dh, k, yaw(90f), nv21(0, 128, 128), w, h)
        val s = c.snapshot()
        val i = 1 * dw + 2 // principal point
        assertEquals(2f, s.xyz[3 * i], 1e-5f); assertEquals(0f, s.xyz[3 * i + 1], 1e-5f); assertEquals(0f, s.xyz[3 * i + 2], 1e-5f)
    }

    @Test fun filtersRangeConfidenceAndInvalidAndDedupesVoxels() {
        val c = DepthCloud(step = 1)
        val d = depth(2000)
        d[0] = 0            // invalid
        d[1] = 100          // 10 cm: too close
        d[2] = 6000         // 6 m: too far
        val conf = ByteArray(dw * dh) { 255.toByte() }
        conf[3] = 10        // low confidence
        assertEquals(dw * dh - 4, c.addKeyframe(d, conf, dw, dh, k, identity(), nv21(0, 128, 128), w, h))
        // the same keyframe again lands in the same voxels: nothing new
        assertEquals(0, c.addKeyframe(d, conf, dw, dh, k, identity(), nv21(0, 128, 128), w, h))
        assertEquals(dw * dh - 4, c.size)
    }

    @Test fun stepSubsamplesAndCapApplies() {
        val c = DepthCloud(maxPoints = 3, step = 1)
        assertEquals(3, c.addKeyframe(depth(1500), null, dw, dh, k, identity(), nv21(0, 128, 128), w, h))
        c.clear()
        assertEquals(0, c.size)
        val c2 = DepthCloud(step = 2)
        assertEquals(2, c2.addKeyframe(depth(1500), null, dw, dh, k, identity(), nv21(0, 128, 128), w, h)) // (0,0),(2,0)
    }

    @Test fun splatLayoutIs32BytesPerPoint() {
        val snap = PointCloudSnapshot(floatArrayOf(1f, 2f, 3f), byteArrayOf(10, 20, 30))
        val bytes = PointCloudSplat.encode(snap)
        assertEquals(32, bytes.size)
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(1f, b.getFloat(0), 0f); assertEquals(2f, b.getFloat(4), 0f); assertEquals(3f, b.getFloat(8), 0f)
        assertEquals(PointCloudSplat.POINT_SCALE_M, b.getFloat(12), 0f)
        assertEquals(10, b.get(24).toInt()); assertEquals(20, b.get(25).toInt()); assertEquals(30, b.get(26).toInt())
        assertEquals(255, b.get(27).toInt() and 0xFF) // opaque
        assertEquals(255, b.get(28).toInt() and 0xFF); assertEquals(128, b.get(29).toInt() and 0xFF) // identity quat
        assertTrue(bytes.size % PointCloudSplat.STRIDE == 0)
    }
}
