package dev.recon3d.capture

import dev.recon3d.capture.ar.pickCameraConfigIndex
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ImageConvertTest {
    /** Builds a plane buffer the way Android does: last row only as long as needed (no trailing padding). */
    private fun plane(w: Int, h: Int, rowStride: Int, pixelStride: Int, value: (x: Int, y: Int) -> Int): ByteArray {
        val size = rowStride * (h - 1) + (w - 1) * pixelStride + 1
        val b = ByteArray(size) { 0x55 } // padding garbage
        for (y in 0 until h) for (x in 0 until w) b[y * rowStride + x * pixelStride] = value(x, y).toByte()
        return b
    }

    @Test fun packsTightPlanarYuvToNv21() {
        val w = 4; val h = 2
        val y = plane(w, h, rowStride = w, pixelStride = 1) { x, yy -> 10 + yy * w + x }
        val u = plane(w / 2, h / 2, rowStride = w / 2, pixelStride = 1) { x, _ -> 100 + x }
        val v = plane(w / 2, h / 2, rowStride = w / 2, pixelStride = 1) { x, _ -> 200 + x }
        val nv21 = ImageConvert.yuv420ToNv21(w, h, y, w, 1, u, v, w / 2, 1)
        val expected = byteArrayOf(10, 11, 12, 13, 14, 15, 16, 17, 200.toByte(), 100, 201.toByte(), 101)
        assertArrayEquals(expected, nv21)
    }

    @Test fun packsPaddedSemiPlanarYuvToNv21() {
        // Typical camera layout: row padding on Y, interleaved chroma (pixelStride 2) with row padding.
        val w = 4; val h = 4
        val yStride = 8; val uvStride = 8
        val y = plane(w, h, yStride, 1) { x, yy -> yy * 16 + x }
        val u = plane(w / 2, h / 2, uvStride, 2) { x, yy -> 100 + yy * 10 + x }
        val v = plane(w / 2, h / 2, uvStride, 2) { x, yy -> 50 + yy * 10 + x }
        val nv21 = ImageConvert.yuv420ToNv21(w, h, y, yStride, 1, u, v, uvStride, 2)
        assertEquals(w * h * 3 / 2, nv21.size)
        val expectedY = byteArrayOf(0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51)
        assertArrayEquals(expectedY, nv21.copyOfRange(0, 16))
        val expectedVu = byteArrayOf(50, 100, 51, 101, 60, 110, 61, 111)
        assertArrayEquals(expectedVu, nv21.copyOfRange(16, 24))
    }

    @Test fun rejectsOddSize() {
        assertThrows(IllegalArgumentException::class.java) {
            ImageConvert.yuv420ToNv21(3, 2, ByteArray(6), 3, 1, ByteArray(2), ByteArray(2), 2, 1)
        }
    }

    @Test fun depth16MasksConfidenceBitsAndHonoursRowStride() {
        // 2x2 depth, rowStride 6 bytes (2 bytes padding), little-endian.
        fun le(v: Int) = byteArrayOf((v and 0xff).toByte(), ((v shr 8) and 0xff).toByte())
        val row0 = le(1000) + le(0xE000 or 8191) + byteArrayOf(9, 9)
        val row1 = le(0) + le(0x2000 or 250)
        val mm = ImageConvert.depth16ToMm(row0 + row1, width = 2, height = 2, rowStrideBytes = 6)
        assertArrayEquals(shortArrayOf(1000, 8191, 0, 250), mm)
    }

    @Test fun picksLargestCpuImageWithLongSideAtMost1920() {
        val dims = listOf(intArrayOf(640, 480), intArrayOf(1920, 1080), intArrayOf(4032, 3024), intArrayOf(1440, 1080))
        assertEquals(1, pickCameraConfigIndex(dims))
    }

    @Test fun fallsBackToSmallestWhenAllExceedLimit() {
        assertEquals(1, pickCameraConfigIndex(listOf(intArrayOf(4032, 3024), intArrayOf(2560, 1440))))
        assertEquals(-1, pickCameraConfigIndex(emptyList()))
    }
}
