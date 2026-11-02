package dev.recon3d.capture

import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer

/** Immutable copy of the accumulated cloud: [xyz] is 3 floats per point (world frame), [gray] 0..255 per point. */
class PointCloudSnapshot(val xyz: FloatArray, val gray: ByteArray) {
    val size: Int get() = gray.size
}

/**
 * Accumulates ARCore feature points (`Frame.acquirePointCloud()`) over a capture, de-duplicated by
 * point id, with an optional grey level sampled from the camera image of a keyframe that saw them.
 * Thread-safe: `add`/`colorize` run on the GL thread, `snapshot`/`clear` from wherever the capture
 * is finished. Pure Kotlin (buffers only), so it is unit-tested.
 */
class PointCloudAccumulator(
    private val maxPoints: Int = MAX_POINTS,
    private val minConfidence: Float = MIN_CONFIDENCE,
) {
    private val index = HashMap<Int, Int>() // point id -> slot
    private var xyz = FloatArray(3 * 1024)
    private var gray = ByteArray(1024)
    private var colored = BooleanArray(1024)
    private var n = 0
    private var lastTimestamp = Long.MIN_VALUE

    val size: Int @Synchronized get() = n

    /**
     * Adds the points of one ARCore point cloud: [points] holds 4 floats per point (x, y, z, confidence),
     * [ids] one id per point. A cloud with the same [timestamp] as the previous call is skipped (ARCore
     * repeats the cloud across frames). Points below [minConfidence] are ignored; a known id updates its
     * position; new ids beyond [maxPoints] are dropped.
     */
    @Synchronized
    fun add(points: FloatBuffer, ids: IntBuffer, timestamp: Long) {
        if (timestamp == lastTimestamp) return
        lastTimestamp = timestamp
        val count = minOf(points.remaining() / 4, ids.remaining())
        val pb = points.position()
        val ib = ids.position()
        for (i in 0 until count) {
            val conf = points.get(pb + 4 * i + 3)
            if (conf < minConfidence) continue
            val id = ids.get(ib + i)
            var slot = index[id] ?: -1
            if (slot < 0) {
                if (n >= maxPoints) continue
                slot = n++
                ensureCapacity(n)
                index[id] = slot
                gray[slot] = DEFAULT_GRAY.toByte()
                colored[slot] = false
            }
            xyz[3 * slot] = points.get(pb + 4 * i)
            xyz[3 * slot + 1] = points.get(pb + 4 * i + 1)
            xyz[3 * slot + 2] = points.get(pb + 4 * i + 2)
        }
    }

    /**
     * Gives every still-uncoloured point that projects into this keyframe the luma of the pixel it lands on.
     * [poseGl] is the column-major world_from_camera matrix (OpenGL camera: looks down -Z, +Y up); the
     * intrinsics are in pixels of the [width]x[height] CPU image whose Y plane is [y] with the given strides.
     */
    @Synchronized
    fun colorize(
        poseGl: FloatArray, fx: Float, fy: Float, cx: Float, cy: Float,
        width: Int, height: Int, y: ByteBuffer, rowStride: Int, pixelStride: Int,
    ) {
        // Rigid inverse: R^T and -R^T t. Column-major: column c of R is poseGl[4c .. 4c+2].
        val r00 = poseGl[0]; val r10 = poseGl[1]; val r20 = poseGl[2]
        val r01 = poseGl[4]; val r11 = poseGl[5]; val r21 = poseGl[6]
        val r02 = poseGl[8]; val r12 = poseGl[9]; val r22 = poseGl[10]
        val tx = poseGl[12]; val ty = poseGl[13]; val tz = poseGl[14]
        for (i in 0 until n) {
            if (colored[i]) continue
            val px = xyz[3 * i] - tx
            val py = xyz[3 * i + 1] - ty
            val pz = xyz[3 * i + 2] - tz
            // camera-frame coordinates (OpenGL)
            val xc = r00 * px + r10 * py + r20 * pz
            val yc = r01 * px + r11 * py + r21 * pz
            val zc = r02 * px + r12 * py + r22 * pz
            if (zc >= -MIN_DEPTH_M) continue // behind or too close
            val u = (fx * xc / -zc + cx).toInt()
            val v = (fy * -yc / -zc + cy).toInt()
            if (u < 0 || u >= width || v < 0 || v >= height) continue
            gray[i] = y.get(v * rowStride + u * pixelStride)
            colored[i] = true
        }
    }

    @Synchronized
    fun snapshot(): PointCloudSnapshot = PointCloudSnapshot(xyz.copyOf(3 * n), gray.copyOf(n))

    @Synchronized
    fun clear() {
        index.clear(); n = 0; lastTimestamp = Long.MIN_VALUE
    }

    private fun ensureCapacity(needed: Int) {
        if (gray.size >= needed) return
        val cap = maxOf(needed, gray.size * 2)
        xyz = xyz.copyOf(3 * cap); gray = gray.copyOf(cap); colored = colored.copyOf(cap)
    }

    companion object {
        const val MAX_POINTS = 200_000
        const val MIN_CONFIDENCE = 0.1f
        /** Grey level of points no keyframe has seen yet. */
        const val DEFAULT_GRAY = 200
        private const val MIN_DEPTH_M = 0.05f
    }
}
