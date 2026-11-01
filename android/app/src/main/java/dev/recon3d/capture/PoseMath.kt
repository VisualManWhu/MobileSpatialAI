package dev.recon3d.capture

import kotlin.math.asin
import kotlin.math.atan2

object PoseMath {
    /**
     * poseGl: 16 floats column-major world_from_camera (OpenGL convention, camera looks down -Z).
     * Returns (yaw, pitch) in degrees: yaw in -180..180 (right turn positive), pitch positive when looking up.
     */
    fun yawPitchDeg(poseGl: FloatArray): Pair<Float, Float> {
        val fx = -poseGl[8].toDouble()
        val fy = -poseGl[9].toDouble()
        val fz = -poseGl[10].toDouble()
        val yaw = Math.toDegrees(atan2(fx, -fz))
        val pitch = Math.toDegrees(asin(fy.coerceIn(-1.0, 1.0)))
        return yaw.toFloat() to pitch.toFloat()
    }
}
