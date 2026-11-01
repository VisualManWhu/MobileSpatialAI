package dev.recon3d.capture

import org.junit.Assert.assertEquals
import org.junit.Test

class PoseMathTest {
    @Test fun identityPoseLooksForward() {
        val (y, p) = PoseMath.yawPitchDeg(identity()); assertEquals(0f, y, 1e-3f); assertEquals(0f, p, 1e-3f)
    }
    @Test fun rightTurnIsPositiveYaw() {
        val (y, p) = PoseMath.yawPitchDeg(pose(90f)); assertEquals(90f, y, 1e-2f); assertEquals(0f, p, 1e-2f)
    }
    @Test fun yawWrapsIntoPlusMinus180() {
        assertEquals(-90f, PoseMath.yawPitchDeg(pose(270f)).first, 1e-2f)
    }
    @Test fun lookUpIsPositivePitch() {
        val (_, p) = PoseMath.yawPitchDeg(pose(30f, 45f)); assertEquals(45f, p, 1e-2f)
        assertEquals(-45f, PoseMath.yawPitchDeg(pose(0f, -45f)).second, 1e-2f)
    }
}
