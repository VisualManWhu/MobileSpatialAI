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

    @Test fun doesNotAliasCallerPoseArray() {
        val sel = KeyframeSelector()
        val buf = identity()
        assertTrue(sel.shouldKeep(0, buf, true))
        buf[12] = 5f // ARCore-style buffer reuse
        // vs the original (x=0) this is only 0.01 m away -> must be dropped
        assertFalse(sel.shouldKeep(1000 * ms, translate(0.01f), true))
    }

    @Test fun translationExactlyAtThresholdIsKept() {
        val sel = KeyframeSelector(minTranslationM = 0.5f)
        assertTrue(sel.shouldKeep(0, identity(), true))
        assertTrue(sel.shouldKeep(1000 * ms, translate(0.5f), true))
    }

    @Test fun intervalExactlyAtThresholdIsKept() {
        val sel = KeyframeSelector()
        assertTrue(sel.shouldKeep(0, identity(), true))
        assertTrue(sel.shouldKeep(200 * ms, translate(1f), true))
    }

    @Test fun firstFrameAfterTrackingLossIsVelocityExempt() {
        val sel = KeyframeSelector()
        assertTrue(sel.shouldKeep(0, identity(), true))
        assertFalse(sel.shouldKeep(500 * ms, identity(), false))
        // 90 deg in 0.1 s after the lost frame would trip the gate if seen state survived
        assertTrue(sel.shouldKeep(600 * ms, yaw(90f), true))
    }
}
