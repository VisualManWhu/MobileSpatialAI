package dev.recon3d.capture

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.ByteArrayInputStream
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
}
