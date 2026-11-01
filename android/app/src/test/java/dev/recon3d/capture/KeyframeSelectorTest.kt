package dev.recon3d.capture

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class KeyframeSelectorTest {
    private val ms = 1_000_000L

    @Test fun dropsFramesWhileNotTracking() {
        assertFalse(KeyframeSelector().shouldKeep(0, identity(), tracking = false))
    }

    @Test fun keepsFirstThenNeedsMotion() {
        val sel = KeyframeSelector()
        assertTrue(sel.shouldKeep(0, identity(), true))
        assertFalse(sel.shouldKeep(300 * ms, identity(), true))
        assertTrue(sel.shouldKeep(600 * ms, translate(0.1f), true))
    }

    @Test fun rotationAloneIsEnoughMotion() {
        val sel = KeyframeSelector()
        assertTrue(sel.shouldKeep(0, identity(), true))
        assertFalse(sel.shouldKeep(1000 * ms, yaw(3f), true))
        assertTrue(sel.shouldKeep(2000 * ms, yaw(8f), true))
    }

    @Test fun respectsMinInterval() {
        val sel = KeyframeSelector()
        assertTrue(sel.shouldKeep(0, identity(), true))
        assertFalse(sel.shouldKeep(100 * ms, translate(0.5f), true))
        assertTrue(sel.shouldKeep(250 * ms, translate(0.5f), true))
    }

    @Test fun rejectsFastRotation() {
        val sel = KeyframeSelector()
        assertTrue(sel.shouldKeep(0, identity(), true))
        // 20 deg in 100 ms = 200 deg/s > 60
        assertFalse(sel.shouldKeep(100 * ms, yaw(20f), true))
    }

    @Test fun acceptsSameRotationWhenSlow() {
        val sel = KeyframeSelector()
        assertTrue(sel.shouldKeep(0, identity(), true))
        // 20 deg in 1 s = 20 deg/s
        assertTrue(sel.shouldKeep(1000 * ms, yaw(20f), true))
    }

    @Test fun velocityIsComparedToPreviousSeenFrameNotLastKept() {
        val sel = KeyframeSelector()
        assertTrue(sel.shouldKeep(0, identity(), true))
        assertTrue(sel.shouldKeep(1000 * ms, yaw(20f), true))   // 20 deg/s, kept
        assertFalse(sel.shouldKeep(1300 * ms, yaw(21f), true))  // seen, too little motion
        // vs last kept (20deg): 15 deg over 0.5 s = 30 deg/s, but vs previous seen (21deg): 14 deg over 0.2 s = 70 deg/s
        assertFalse(sel.shouldKeep(1500 * ms, yaw(35f), true))
    }
}
