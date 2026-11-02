package dev.recon3d.capture

/** Pure-Kotlin pixel repacking for ARCore CPU images (byte arrays copied out of android.media.Image planes). */
object ImageConvert {
    /**
     * Packs YUV_420_888 planes into NV21 (full Y plane, then interleaved V,U at half resolution),
     * honouring each plane's row and pixel stride. Width and height must be even.
     */
    fun yuv420ToNv21(
        width: Int, height: Int,
        y: ByteArray, yRowStride: Int, yPixelStride: Int,
        u: ByteArray, v: ByteArray, uvRowStride: Int, uvPixelStride: Int,
    ): ByteArray {
        require(width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0) { "YUV size must be even: ${width}x$height" }
        val out = ByteArray(width * height * 3 / 2)
        var o = 0
        if (yPixelStride == 1) {
            for (r in 0 until height) { System.arraycopy(y, r * yRowStride, out, o, width); o += width }
        } else {
            for (r in 0 until height) {
                val base = r * yRowStride
                for (c in 0 until width) out[o++] = y[base + c * yPixelStride]
            }
        }
        for (r in 0 until height / 2) {
            val base = r * uvRowStride
            for (c in 0 until width / 2) {
                val i = base + c * uvPixelStride
                out[o++] = v[i]
                out[o++] = u[i]
            }
        }
        return out
    }

    /**
     * `Frame.acquireDepthImage16Bits()` plane (little-endian uint16, full 0..65535 mm range; confidence is a
     * separate image) → millimetres, honouring the row stride. Values are kept as the raw uint16 bit pattern
     * in a Short (Png16 writes them unsigned).
     */
    fun depth16ToMm(data: ByteArray, width: Int, height: Int, rowStrideBytes: Int): ShortArray {
        val out = ShortArray(width * height)
        for (r in 0 until height) {
            val base = r * rowStrideBytes
            for (c in 0 until width) {
                val i = base + c * 2
                val raw = (data[i].toInt() and 0xff) or ((data[i + 1].toInt() and 0xff) shl 8)
                out[r * width + c] = raw.toShort()
            }
        }
        return out
    }

    /**
     * Single-byte-per-pixel plane (e.g. Y8 `Frame.acquireRawDepthConfidenceImage()`) → tight `width*height`
     * bytes, dropping the row padding when [rowStride] > [width].
     */
    fun plane8(data: ByteArray, width: Int, height: Int, rowStride: Int): ByteArray {
        require(rowStride >= width) { "rowStride $rowStride < width $width" }
        val out = ByteArray(width * height)
        for (r in 0 until height) System.arraycopy(data, r * rowStride, out, r * width, width)
        return out
    }
}
