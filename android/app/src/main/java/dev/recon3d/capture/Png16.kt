package dev.recon3d.capture

import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.util.zip.CRC32
import java.util.zip.Deflater

/** Minimal 16-bit grayscale PNG encoder (Android Bitmap cannot write 16-bit PNGs). */
object Png16 {
    private val SIGNATURE = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)

    /** [pixels] row-major, size width*height, treated as unsigned 0..65535. */
    fun encode(width: Int, height: Int, pixels: ShortArray): ByteArray {
        require(width > 0 && height > 0 && pixels.size == width * height) { "pixel count != width*height" }
        val raw = ByteArray(height * (1 + width * 2))
        var o = 0
        for (y in 0 until height) {
            raw[o++] = 0 // filter type 0 (None)
            for (x in 0 until width) {
                val v = pixels[y * width + x].toInt()
                raw[o++] = (v ushr 8).toByte()
                raw[o++] = v.toByte()
            }
        }
        val out = ByteArrayOutputStream(raw.size / 2 + 64)
        out.write(SIGNATURE)
        val hdr = ByteArrayOutputStream(13)
        DataOutputStream(hdr).apply {
            writeInt(width); writeInt(height)
            writeByte(16) // bit depth
            writeByte(0)  // color type: grayscale
            writeByte(0); writeByte(0); writeByte(0) // compression, filter, interlace
        }
        chunk(out, "IHDR", hdr.toByteArray())
        chunk(out, "IDAT", deflate(raw))
        chunk(out, "IEND", ByteArray(0))
        return out.toByteArray()
    }

    private fun deflate(data: ByteArray): ByteArray {
        val d = Deflater(6)
        try {
            d.setInput(data); d.finish()
            val bos = ByteArrayOutputStream(data.size / 2 + 16)
            val buf = ByteArray(8192)
            while (!d.finished()) bos.write(buf, 0, d.deflate(buf))
            return bos.toByteArray()
        } finally { d.end() }
    }

    private fun chunk(out: ByteArrayOutputStream, type: String, data: ByteArray) {
        val dos = DataOutputStream(out)
        dos.writeInt(data.size)
        val typeBytes = type.toByteArray(Charsets.US_ASCII)
        dos.write(typeBytes); dos.write(data)
        val crc = CRC32().apply { update(typeBytes); update(data) }
        dos.writeInt(crc.value.toInt())
    }
}
