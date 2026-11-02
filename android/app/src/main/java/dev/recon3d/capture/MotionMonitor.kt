package dev.recon3d.capture

import kotlin.math.hypot

/** What the capture overlay shows about the camera: a warning (or none) and whether ARCore is tracking. */
data class CaptureStatus(val warning: String?, val tracking: Boolean)

/**
 * Per-frame camera health for the capture screen: "tracking lost" and "turning too fast" warnings.
 * Pure Kotlin, no Android deps. (Replaces the three-stage guide, which was removed.)
 */
class MotionMonitor {
    private var prevT = 0L
    private var prevYaw = 0f
    private var prevPitch = 0f
    private var havePrev = false

    var last: CaptureStatus = CaptureStatus(null, true)
        private set

    fun onFrame(tNs: Long, poseGl: FloatArray, tracking: Boolean): CaptureStatus {
        if (!tracking) {
            havePrev = false // no velocity across a tracking gap
            last = CaptureStatus(MSG_LOST, false)
            return last
        }
        val (yaw, pitch) = PoseMath.yawPitchDeg(poseGl)
        var warning: String? = null
        if (havePrev) {
            val dt = (tNs - prevT) / 1e9
            if (dt > 0 && hypot(wrap180(yaw - prevYaw).toDouble(), (pitch - prevPitch).toDouble()) / dt > MAX_DEG_PER_S) {
                warning = MSG_TOO_FAST
            }
        }
        prevT = tNs; prevYaw = yaw; prevPitch = pitch; havePrev = true
        last = CaptureStatus(warning, true)
        return last
    }

    private fun wrap180(d: Float): Float {
        var r = ((d + 180f) % 360f + 360f) % 360f - 180f
        if (r == -180f) r = 180f
        return r
    }

    companion object {
        const val MSG_LOST = "跟丢了，请回到刚才的位置"
        const val MSG_TOO_FAST = "太快了，慢一点"
        /** Above this angular speed the capture is likely motion-blurred (Pixel 10 Pro, indoor light). */
        const val MAX_DEG_PER_S = 30.0
    }
}
