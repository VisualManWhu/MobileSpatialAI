package dev.recon3d.capture

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.zip.Inflater
import javax.imageio.ImageIO

class Png16Test {
    @Test fun roundTripsViaImageIo() {
        val px = shortArrayOf(0, 1000, 65535.toShort(), 2000, 7, 40000.toShort())
        val img = ImageIO.read(ByteArrayInputStream(Png16.encode(3, 2, px)))
        assertEquals(3, img.width); assertEquals(2, img.height)
        assertEquals(16, img.colorModel.getComponentSize(0))
        val raster = img.raster
        val out = ShortArray(6) { raster.getSample(it % 3, it / 3, 0).toShort() }
        assertArrayEquals(px, out)
    }

    @Test fun headerFields() {
        val b = Png16.encode(2, 2, ShortArray(4))
        assertEquals(0x89, b[0].toInt() and 0xff)
        assertEquals(16, b[24].toInt()); assertEquals(0, b[25].toInt())
    }

    @Test fun png8HeaderAndPixels() {
        val b = Png8.encode(2, 1, byteArrayOf(0, 255.toByte()))
        assertEquals(0x89, b[0].toInt() and 0xff)
        // IHDR data starts at 16: width(4) height(4) bitDepth colourType
        assertEquals(2, java.nio.ByteBuffer.wrap(b, 16, 4).int)
        assertEquals(1, java.nio.ByteBuffer.wrap(b, 20, 4).int)
        assertEquals(8, b[24].toInt()); assertEquals(0, b[25].toInt())
        // IDAT follows IHDR (8 sig + 25 IHDR chunk): length at 33, type at 37, data at 41
        val idatLen = java.nio.ByteBuffer.wrap(b, 33, 4).int
        assertEquals("IDAT", String(b, 37, 4, Charsets.US_ASCII))
        val inf = Inflater().apply { setInput(b, 41, idatLen) }
        val out = ByteArrayOutputStream(); val buf = ByteArray(64)
        while (!inf.finished()) out.write(buf, 0, inf.inflate(buf))
        inf.end()
        // one row: filter byte 0 then pixels
        assertArrayEquals(byteArrayOf(0, 0, 255.toByte()), out.toByteArray())
    }

    @Test fun png8RoundTripsViaImageIo() {
        val px = byteArrayOf(0, 17, 200.toByte(), 255.toByte(), 1, 128.toByte())
        val img = ImageIO.read(ByteArrayInputStream(Png8.encode(3, 2, px)))
        assertEquals(3, img.width); assertEquals(2, img.height)
        assertEquals(8, img.colorModel.getComponentSize(0))
        val out = ByteArray(6) { img.raster.getSample(it % 3, it / 3, 0).toByte() }
        assertArrayEquals(px, out)
    }
}
