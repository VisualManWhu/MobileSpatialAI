package dev.recon3d.capture

/** Immutable copy of the preview cloud: [xyz] is 3 floats per point (world frame), [rgb] 3 bytes per point. */
class PointCloudSnapshot(val xyz: FloatArray, val rgb: ByteArray) {
    val size: Int get() = rgb.size / 3
}

/**
 * Dense, coloured preview cloud built from the keyframes' depth maps: every [step]-th depth pixel in
 * 0.3..3.5 m (and, when a raw-depth confidence map exists, confidence >= [minConfidence]) is back-projected
 * with the depth intrinsics and the frame's pose and binned into 2 cm voxels. A voxel becomes a point
 * (coloured from the NV21 camera image) only once [minViews] different keyframes have hit it: ARCore depth
 * disagrees with itself by 5–20 cm between frames, and single-frame slabs would otherwise litter the view.
 * Pure Kotlin; called from the pipeline's single IO worker, so the only synchronisation is around
 * [snapshot]/[clear] versus [addKeyframe].
 */
class DepthCloud(
    private val maxPoints: Int = MAX_POINTS,
    private val step: Int = STEP,
    private val voxelM: Float = VOXEL_M,
    private val minConfidence: Int = MIN_CONFIDENCE,
    private val minViews: Int = MIN_VIEWS,
) {
    /** voxel -> keyframes that hit it so far (counts stop at [minViews]; a value of -1 = already a point). */
    private val voxels = HashMap<Long, Int>()
    private val hitThisFrame = HashSet<Long>()
    private var xyz = FloatArray(3 * 4096)
    private var rgb = ByteArray(3 * 4096)
    private var n = 0

    val size: Int @Synchronized get() = n

    /**
     * Adds one keyframe. [depthMm] is dw*dh uint16 millimetres (0 = invalid) as raw Short bit patterns;
     * [confidence] is dw*dh bytes 0..255 or null; [k] = (fx, fy, cx, cy) in depth pixels; [poseGl] the
     * column-major world_from_camera of the depth (OpenGL camera: looks down -Z, +Y up); [nv21] the
     * width x height camera image (full-resolution colour source; depth pixel (u, v) maps to image pixel
     * (u * width / dw, v * height / dh)). Returns how many points were added.
     */
    @Synchronized
    fun addKeyframe(
        depthMm: ShortArray, confidence: ByteArray?, dw: Int, dh: Int, k: FloatArray, poseGl: FloatArray,
        nv21: ByteArray, width: Int, height: Int,
    ): Int {
        require(k.size == 4 && poseGl.size == 16)
        val fx = k[0]; val fy = k[1]; val cx = k[2]; val cy = k[3]
        val sx = width.toFloat() / dw; val sy = height.toFloat() / dh
        var added = 0
        hitThisFrame.clear()
        var v = 0
        while (v < dh) {
            var u = 0
            while (u < dw) {
                val i = v * dw + u
                val mm = depthMm[i].toInt() and 0xFFFF
                val z = mm / 1000f
                val ok = mm != 0 && z >= MIN_DEPTH_M && z <= MAX_DEPTH_M &&
                    (confidence == null || (confidence[i].toInt() and 0xFF) >= minConfidence)
                if (ok && n < maxPoints) {
                    // camera frame (OpenGL): x right, y up, z = -depth
                    val xc = (u - cx) / fx * z
                    val yc = -(v - cy) / fy * z
                    val zc = -z
                    val wx = poseGl[0] * xc + poseGl[4] * yc + poseGl[8] * zc + poseGl[12]
                    val wy = poseGl[1] * xc + poseGl[5] * yc + poseGl[9] * zc + poseGl[13]
                    val wz = poseGl[2] * xc + poseGl[6] * yc + poseGl[10] * zc + poseGl[14]
                    val key = voxelKey(wx, wy, wz)
                    val seen = voxels[key] ?: 0
                    // -1 = already a point; a keyframe counts at most once per voxel
                    if (seen >= 0 && hitThisFrame.add(key)) {
                        val views = seen + 1
                        if (views >= minViews) {
                            voxels[key] = -1
                            ensureCapacity(n + 1)
                            xyz[3 * n] = wx; xyz[3 * n + 1] = wy; xyz[3 * n + 2] = wz
                            val px = (u * sx).toInt().coerceIn(0, width - 1)
                            val py = (v * sy).toInt().coerceIn(0, height - 1)
                            nv21Rgb(nv21, width, height, px, py, rgb, 3 * n)
                            n++; added++
                        } else {
                            voxels[key] = views
                        }
                    }
                }
                u += step
            }
            v += step
        }
        return added
    }

    @Synchronized
    fun snapshot(): PointCloudSnapshot = PointCloudSnapshot(xyz.copyOf(3 * n), rgb.copyOf(3 * n))

    @Synchronized
    fun clear() { voxels.clear(); hitThisFrame.clear(); n = 0 }

    private fun voxelKey(x: Float, y: Float, z: Float): Long {
        val ix = Math.floorDiv((x * 1000).toInt(), (voxelM * 1000).toInt()).toLong() and 0x1FFFFF
        val iy = Math.floorDiv((y * 1000).toInt(), (voxelM * 1000).toInt()).toLong() and 0x1FFFFF
        val iz = Math.floorDiv((z * 1000).toInt(), (voxelM * 1000).toInt()).toLong() and 0x1FFFFF
        return (ix shl 42) or (iy shl 21) or iz
    }

    private fun ensureCapacity(needed: Int) {
        if (rgb.size / 3 >= needed) return
        val cap = maxOf(needed, rgb.size / 3 * 2)
        xyz = xyz.copyOf(3 * cap); rgb = rgb.copyOf(3 * cap)
    }

    companion object {
        const val MAX_POINTS = 500_000
        /** Sample every 2nd depth pixel in both axes (160x90 -> 80x45 per keyframe). */
        const val STEP = 2
        const val VOXEL_M = 0.02f
        /** Raw-depth confidence threshold (0..255): stricter than the server's 128, this is only a preview. */
        const val MIN_CONFIDENCE = 160
        /** Keyframes that must agree on a voxel before it is shown. */
        const val MIN_VIEWS = 2
        const val MIN_DEPTH_M = 0.3f
        const val MAX_DEPTH_M = 3.5f

        /** BT.601 full-range YUV -> RGB of NV21 pixel ([px], [py]) into [out] at [o]. */
        fun nv21Rgb(nv21: ByteArray, width: Int, height: Int, px: Int, py: Int, out: ByteArray, o: Int) {
            val yv = nv21[py * width + px].toInt() and 0xFF
            val uvBase = width * height + (py / 2) * width + (px / 2) * 2
            val vv = (nv21[uvBase].toInt() and 0xFF) - 128
            val uu = (nv21[uvBase + 1].toInt() and 0xFF) - 128
            out[o] = (yv + 1.402f * vv).toInt().coerceIn(0, 255).toByte()
            out[o + 1] = (yv - 0.344136f * uu - 0.714136f * vv).toInt().coerceIn(0, 255).toByte()
            out[o + 2] = (yv + 1.772f * uu).toInt().coerceIn(0, 255).toByte()
        }
    }
}
