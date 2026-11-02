package dev.recon3d.capture

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer

class PointCloudTest {
    private fun cloud(vararg pts: FloatArray): Pair<FloatBuffer, IntBuffer> {
        // each entry: id, x, y, z, confidence
        val f = FloatBuffer.allocate(4 * pts.size)
        val ids = IntBuffer.allocate(pts.size)
        for (p in pts) { ids.put(p[0].toInt()); f.put(p[1]); f.put(p[2]); f.put(p[3]); f.put(p[4]) }
        f.flip(); ids.flip()
        return f to ids
    }

    @Test fun dedupesByIdFiltersConfidenceAndSkipsRepeatedTimestamp() {
        val acc = PointCloudAccumulator()
        val (f1, i1) = cloud(floatArrayOf(7f, 1f, 2f, 3f, 0.9f), floatArrayOf(8f, 0f, 0f, 0f, 0.05f))
        acc.add(f1, i1, timestamp = 100)
        assertEquals(1, acc.size) // id 8 below confidence
        val (f2, i2) = cloud(floatArrayOf(7f, 1.5f, 2f, 3f, 0.9f), floatArrayOf(9f, 4f, 5f, 6f, 0.5f))
        acc.add(f2, i2, timestamp = 100) // same cloud again: ignored
        assertEquals(1, acc.size)
        acc.add(f2, i2, timestamp = 200)
        assertEquals(2, acc.size)
        val s = acc.snapshot()
        assertEquals(1.5f, s.xyz[0], 0f) // id 7 updated to its latest position
        assertEquals(4f, s.xyz[3], 0f)
        assertEquals(PointCloudAccumulator.DEFAULT_GRAY, s.gray[0].toInt() and 0xFF)
    }

    @Test fun capsNewPoints() {
        val acc = PointCloudAccumulator(maxPoints = 2)
        val (f, i) = cloud(floatArrayOf(1f, 0f, 0f, 0f, 1f), floatArrayOf(2f, 0f, 0f, 0f, 1f), floatArrayOf(3f, 0f, 0f, 0f, 1f))
        acc.add(f, i, 1)
        assertEquals(2, acc.size)
    }

    @Test fun colorizeSamplesLumaOfProjectedPixelOnlyForPointsInFront() {
        val acc = PointCloudAccumulator()
        // point A 2 m straight ahead of a camera at the origin (OpenGL: ahead = -Z); point B behind it
        val (f, i) = cloud(floatArrayOf(1f, 0f, 0f, -2f, 1f), floatArrayOf(2f, 0f, 0f, 2f, 1f))
        acc.add(f, i, 1)
        val w = 8; val h = 6; val rowStride = 10; val pixelStride = 1
        val y = ByteBuffer.allocate(rowStride * h)
        val cx = 4f; val cy = 3f
        y.put(3 * rowStride + 4, 200.toByte())
        acc.colorize(identity(), fx = 10f, fy = 10f, cx = cx, cy = cy, width = w, height = h, y = y, rowStride = rowStride, pixelStride = pixelStride)
        val s = acc.snapshot()
        assertEquals(200, s.gray[0].toInt() and 0xFF)
        assertEquals(PointCloudAccumulator.DEFAULT_GRAY, s.gray[1].toInt() and 0xFF)
        // a later keyframe does not recolour an already coloured point
        y.put(3 * rowStride + 4, 50.toByte())
        acc.colorize(identity(), 10f, 10f, cx, cy, w, h, y, rowStride, pixelStride)
        assertEquals(200, acc.snapshot().gray[0].toInt() and 0xFF)
    }

    @Test fun colorizeUsesCameraPoseAndImageAxes() {
        val acc = PointCloudAccumulator()
        // camera at x=1 looking down -Z; a point 1 m ahead and 0.5 m to the camera's right, 0.25 m up
        val (f, i) = cloud(floatArrayOf(1f, 1.5f, 0.25f, -1f, 1f))
        acc.add(f, i, 1)
        val w = 100; val h = 100; val y = ByteBuffer.allocate(w * h)
        // fx = fy = 40, principal point (50, 50): expected u = 50 + 40*0.5 = 70, v = 50 - 40*0.25 = 40 (y down)
        y.put(40 * w + 70, 123.toByte())
        acc.colorize(translate(1f), 40f, 40f, 50f, 50f, w, h, y, rowStride = w, pixelStride = 1)
        assertEquals(123, acc.snapshot().gray[0].toInt() and 0xFF)
    }

    @Test fun splatLayoutIs32BytesPerPoint() {
        val snap = PointCloudSnapshot(floatArrayOf(1f, 2f, 3f), byteArrayOf(77))
        val bytes = PointCloudSplat.encode(snap)
        assertEquals(32, bytes.size)
        val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(1f, b.getFloat(0), 0f); assertEquals(2f, b.getFloat(4), 0f); assertEquals(3f, b.getFloat(8), 0f)
        assertEquals(PointCloudSplat.POINT_SCALE_M, b.getFloat(12), 0f)
        assertEquals(77, b.get(24).toInt() and 0xFF); assertEquals(77, b.get(26).toInt() and 0xFF)
        assertEquals(255, b.get(27).toInt() and 0xFF) // opaque
        assertEquals(255, b.get(28).toInt() and 0xFF); assertEquals(128, b.get(29).toInt() and 0xFF) // identity quat
        assertEquals(0, bytes.size % PointCloudSplat.STRIDE)
    }
}
