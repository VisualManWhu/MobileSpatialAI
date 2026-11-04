package dev.recon3d.viewer

import dev.recon3d.capture.CapturePackageWriter
import dev.recon3d.capture.pose
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class HomeViewTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun `centre is the mean camera position, look is the first frame's forward`() {
        val h = HomeView.fromPoses(listOf(pose(90f, x = 0f, y = 1f, z = 0f), pose(0f, x = 2f, y = 1f, z = -4f), pose(180f, x = 4f, y = 1.6f, z = 1f)))!!
        assertEquals(2f, h.x, 1e-6f); assertEquals(1.2f, h.y, 1e-6f); assertEquals(-1f, h.z, 1e-6f)
        // yaw 90 degrees = looking +X
        assertEquals(1f, h.lx, 1e-6f); assertEquals(0f, h.ly, 1e-6f); assertEquals(0f, h.lz, 1e-6f)
    }

    @Test fun `vertical first frame falls back to -Z, empty list is null`() {
        val h = HomeView.fromPoses(listOf(pose(0f, pitchDeg = -90f, x = 1f)))!!
        assertEquals(0f, h.lx, 1e-6f); assertEquals(-1f, h.lz, 1e-6f)
        assertNull(HomeView.fromPoses(emptyList()))
    }

    @Test fun `query uses dot decimals and four places`() {
        assertEquals("&home=1.5000,-0.2500,3.0000&look=0.0000,0.0000,-1.0000", HomeView(1.5f, -0.25f, 3f, 0f, 0f, -1f).query())
    }

    @Test fun `reads the poses back from a capture package`() {
        val w = CapturePackageWriter(File(tmp.root, "stage"), "room-1", "cap-1", 64, 48, 4, 3, device = "TestPhone")
        w.addFrame(1, ByteArray(8), null, pose(0f, x = 0f, y = 1f, z = 0f), 500f, 500f, 320f, 240f)
        w.addFrame(2, ByteArray(8), null, pose(0f, x = 2f, y = 1f, z = -2f), 500f, 500f, 320f, 240f)
        val h = HomeView.fromCaptureZip(w.finish())!!
        assertEquals(1f, h.x, 1e-5f); assertEquals(1f, h.y, 1e-5f); assertEquals(-1f, h.z, 1e-5f)
        assertEquals(-1f, h.lz, 1e-5f)
    }

    @Test fun `missing zip gives null`() {
        assertNull(HomeView.fromCaptureZip(File(tmp.root, "nope.zip")))
    }
}
