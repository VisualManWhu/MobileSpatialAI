package dev.recon3d.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MotionMonitorTest {
    private val ms = 1_000_000L

    @Test fun trackingLostWarns() {
        val m = MotionMonitor()
        m.onFrame(0, yaw(0f), tracking = true)
        val s = m.onFrame(33 * ms, yaw(0f), tracking = false)
        assertEquals("跟丢了，请回到刚才的位置", s.warning)
        assertFalse(s.tracking)
    }

    @Test fun fastRotationWarns() {
        val m = MotionMonitor()
        m.onFrame(0, yaw(0f), tracking = true)
        // 4 deg in 100 ms = 40 deg/s > 30
        assertEquals("太快了，慢一点", m.onFrame(100 * ms, yaw(4f), tracking = true).warning)
    }

    @Test fun slowRotationNoWarning() {
        val m = MotionMonitor()
        m.onFrame(0, yaw(0f), tracking = true)
        // 2 deg in 100 ms = 20 deg/s
        val s = m.onFrame(100 * ms, yaw(2f), tracking = true)
        assertNull(s.warning)
        assertTrue(s.tracking)
    }

    @Test fun firstFrameAfterLossNeverWarnsFast() {
        val m = MotionMonitor()
        m.onFrame(0, yaw(0f), tracking = true)
        m.onFrame(33 * ms, yaw(0f), tracking = false)
        // Big jump right after re-acquiring tracking is not a "too fast" (no valid previous frame).
        assertNull(m.onFrame(66 * ms, yaw(90f), tracking = true).warning)
    }
}
