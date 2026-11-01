package dev.recon3d.capture

import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Decides which camera frames become keyframes.
 * Motion thresholds and interval compare against the last KEPT frame;
 * the angular-velocity gate compares against the previous SEEN (tracking) frame.
 */
class KeyframeSelector(
    private val minIntervalNs: Long = 200_000_000,
    private val minTranslationM: Float = 0.05f,
    private val minRotationDeg: Float = 5f,
    private val maxAngularVelocityDegPerS: Float = 60f,
) {
    private var keptT = 0L
    private var keptPose: FloatArray? = null
    private var seenT = 0L
    private var seenPose: FloatArray? = null

    fun shouldKeep(tNs: Long, poseGl: FloatArray, tracking: Boolean): Boolean {
        if (!tracking) { seenPose = null; return false }
        val prevSeen = seenPose
        val prevSeenT = seenT
        seenPose = poseGl.copyOf(); seenT = tNs

        if (prevSeen != null && tNs > prevSeenT) {
            val dtS = (tNs - prevSeenT) / 1e9
            if (rotationDeg(prevSeen, poseGl) / dtS > maxAngularVelocityDegPerS) return false
        }
        val kept = keptPose ?: return keep(tNs, poseGl)
        if (tNs - keptT < minIntervalNs) return false  // elapsed >= minInterval passes
        if (translationM(kept, poseGl) >= minTranslationM || rotationDeg(kept, poseGl) >= minRotationDeg) return keep(tNs, poseGl)
        return false
    }

    private fun keep(tNs: Long, p: FloatArray): Boolean { keptT = tNs; keptPose = p.copyOf(); return true }

    companion object {
        fun translationM(a: FloatArray, b: FloatArray): Float {
            val dx = a[12] - b[12]; val dy = a[13] - b[13]; val dz = a[14] - b[14]
            return sqrt(dx * dx + dy * dy + dz * dz)
        }

        /** angle = acos((trace(Ra^T Rb) - 1) / 2); trace = sum of element-wise products of the 3x3 blocks. */
        fun rotationDeg(a: FloatArray, b: FloatArray): Float {
            var tr = 0.0
            for (c in 0 until 3) for (r in 0 until 3) tr += a[c * 4 + r].toDouble() * b[c * 4 + r]
            val cos = ((tr - 1.0) / 2.0).coerceIn(-1.0, 1.0)
            return Math.toDegrees(acos(cos)).toFloat()
        }
    }
}
