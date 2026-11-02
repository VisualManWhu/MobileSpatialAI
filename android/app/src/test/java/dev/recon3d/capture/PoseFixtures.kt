package dev.recon3d.capture

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** Shared test pose builders: 16 floats, column-major world_from_camera (OpenGL: camera looks down -Z, +Y up). */
private var clockNs = 0L
fun nextTns(dtMs: Long = 200): Long { clockNs += dtMs * 1_000_000L; return clockNs }

/** yaw: right turn positive; pitch: look-up positive. Position at (x,y,z). */
fun pose(yawDeg: Float, pitchDeg: Float = 0f, x: Float = 0f, y: Float = 0f, z: Float = 0f): FloatArray {
    val yr = Math.toRadians(yawDeg.toDouble())
    val pr = Math.toRadians(pitchDeg.toDouble())
    val fx = cos(pr) * sin(yr); val fy = sin(pr); val fz = -cos(pr) * cos(yr)
    val bx = -fx; val by = -fy; val bz = -fz
    // right = up_world x back
    var rx = bz; val ry = 0.0; var rz = -bx
    val rl = sqrt(rx * rx + rz * rz)
    if (rl < 1e-9) { rx = 1.0; rz = 0.0 } else { rx /= rl; rz /= rl }
    // up = back x right
    val ux = by * rz - bz * ry; val uy = bz * rx - bx * rz; val uz = bx * ry - by * rx
    return floatArrayOf(
        rx.toFloat(), ry.toFloat(), rz.toFloat(), 0f,
        ux.toFloat(), uy.toFloat(), uz.toFloat(), 0f,
        bx.toFloat(), by.toFloat(), bz.toFloat(), 0f,
        x, y, z, 1f,
    )
}

fun identity(): FloatArray = pose(0f)
fun yaw(deg: Float): FloatArray = pose(deg)
fun translate(x: Float, y: Float = 0f, z: Float = 0f): FloatArray = pose(0f, 0f, x, y, z)
