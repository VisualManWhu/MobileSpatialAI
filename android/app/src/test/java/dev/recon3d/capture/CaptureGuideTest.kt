package dev.recon3d.capture

import org.junit.Assert.*
import org.junit.Test

class CaptureGuideTest {
    @Test fun fullSpinCompletesStage() { val g = CaptureGuide(); spin(g, 0..355 step 5); assertEquals(Stage.SWEEP, g.last.stage) }
    @Test fun cannotFinishBeforeSpin() { val g = CaptureGuide(); spin(g, 0..90 step 5); assertFalse(g.last.canFinish); assertTrue(g.last.hint.contains("先完成原地转圈")) }
    @Test fun trackingLostDoesNotAddCoverage() { val g = CaptureGuide(); spin(g, 0..355 step 5, tracking = false); assertEquals(0f, g.last.stageProgress); assertEquals(Stage.SPIN, g.last.stage); assertEquals("跟丢了，请回到刚才的位置", g.last.warning); assertEquals(Arrow.NONE, g.last.arrow) }
    @Test fun fastRotationWarns() { val g = CaptureGuide(); g.onFrame(0, yaw(0f), true); val ui = g.onFrame(100_000_000, yaw(20f), true); assertEquals("太快了，慢一点", ui.warning) }
    @Test fun above30DegPerSecWarns() { val g = CaptureGuide(); g.onFrame(0, yaw(0f), true); assertEquals("太快了，慢一点", g.onFrame(100_000_000, yaw(4f), true).warning) }
    @Test fun below30DegPerSecNoWarning() { val g = CaptureGuide(); g.onFrame(0, yaw(0f), true); assertNull(g.onFrame(100_000_000, yaw(2f), true).warning) }
    @Test fun firstFrameNeverWarns() { val g = CaptureGuide(); assertNull(g.onFrame(0, yaw(100f), true).warning) }
    @Test fun slowRotationNoWarning() { val g = CaptureGuide(); g.onFrame(0, yaw(0f), true); assertNull(g.onFrame(1_000_000_000, yaw(20f), true).warning) }
    @Test fun sweepNeedsUpAndDown() { val g = afterSpin(); pitchTo(g, 50f); assertEquals(0.5f, g.last.stageProgress); assertEquals(Arrow.DOWN, g.last.arrow) }

    @Test fun initialStateIsSpin() { val u = CaptureGuide().last; assertEquals(Stage.SPIN, u.stage); assertEquals(Arrow.RIGHT, u.arrow); assertFalse(u.canFinish) }
    @Test fun spinWrapAroundCounts() { val g = CaptureGuide(); spin(g, 180..355 step 5); spin(g, 0..175 step 5); assertEquals(Stage.SWEEP, g.last.stage) }
    @Test fun spinIgnoresHighPitch() {
        val g = CaptureGuide(); for (d in 0..355 step 5) feed(g, pose(d.toFloat(), 40f)); assertEquals(Stage.SPIN, g.last.stage); assertEquals(0f, g.last.stageProgress)
    }
    @Test fun spinArrowLeftWhenUncoveredIsLeft() {
        val h = CaptureGuide(); spin(h, 0 downTo -35 step 5)
        assertEquals(Arrow.LEFT, h.last.arrow)
    }
    @Test fun sweepDownFirstStillAsksUp() { val g = afterSpin(); pitchTo(g, -50f); assertEquals(0.5f, g.last.stageProgress); assertEquals(Arrow.UP, g.last.arrow) }
    @Test fun sweepCompletesToRoam() { val g = afterSpin(); pitchTo(g, 50f); pitchTo(g, -50f); assertEquals(Stage.ROAM, g.last.stage) }
    @Test fun skipSweepGoesToRoamThenDone() {
        val g = afterSpin(); g.skipStage(); assertEquals(Stage.ROAM, g.last.stage); g.skipStage(); assertEquals(Stage.DONE, g.last.stage)
    }
    @Test fun skipIgnoredInSpin() { val g = CaptureGuide(); g.skipStage(); assertEquals(Stage.SPIN, g.last.stage) }
    @Test fun doneState() {
        val g = afterSpin(); g.skipStage(); g.skipStage(); val u = g.last
        assertEquals(Arrow.NONE, u.arrow); assertEquals("拍摄完成，可以上传了", u.hint); assertTrue(u.canFinish)
    }
    @Test fun canFinishAfterSpin() { assertTrue(afterSpin().last.canFinish) }

    private fun roamGuide(): CaptureGuide { val g = afterSpin(); g.skipStage(); return g }

    @Test fun roamStaticDoesNotComplete() {
        val g = roamGuide(); for (d in 0 until 360 step 45) feed(g, pose(d.toFloat() - 180f)); assertEquals(Stage.ROAM, g.last.stage); assertEquals(0f, g.last.stageProgress)
    }
    @Test fun roamCompletesWithWalkAndLooking() {
        val g = roamGuide()
        for (i in 1..5) {
            val x = i * 0.5f
            feed(g, translate(x))
            for (k in 0 until 8) feed(g, pose(-180f + 45f * k + 10f, 0f, x), dtMs = 2000)
        }
        assertEquals(Stage.DONE, g.last.stage)
    }
    @Test fun roamArrowPointsToUncoveredSector() {
        val g = roamGuide(); feed(g, translate(1.0f)); feed(g, pose(0f, 0f, 1.0f), dtMs = 2000)
        assertTrue(g.last.arrow in listOf(Arrow.FORWARD_LEFT, Arrow.FORWARD_RIGHT, Arrow.LEFT, Arrow.RIGHT))
        assertEquals("向箭头方向走两步", g.last.hint)
    }
}
