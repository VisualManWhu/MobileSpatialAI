package dev.recon3d.capture.ar

import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import java.util.EnumSet

/** Result of [ArSessionHelper.configure]: whether depth is on, and the chosen CPU image size. */
data class ArSetup(val depthEnabled: Boolean, val imageW: Int, val imageH: Int)

object ArSessionHelper {
    private const val MAX_LONG_SIDE = 1920

    /**
     * Picks the camera config with the largest CPU image (long side ≤ 1920), preferring configs that
     * use a hardware depth sensor, then enables automatic depth (if supported), auto focus and
     * LATEST_CAMERA_IMAGE updates. Call on a paused (e.g. freshly created) session.
     */
    fun configure(session: Session): ArSetup {
        fun configs(usage: CameraConfig.DepthSensorUsage): List<CameraConfig> =
            session.getSupportedCameraConfigs(CameraConfigFilter(session).setDepthSensorUsage(EnumSet.of(usage)))
        val candidates = configs(CameraConfig.DepthSensorUsage.REQUIRE_AND_USE)
            .ifEmpty { configs(CameraConfig.DepthSensorUsage.DO_NOT_USE) }
        val idx = pickCameraConfigIndex(candidates.map { intArrayOf(it.imageSize.width, it.imageSize.height) }, MAX_LONG_SIDE)
        if (idx >= 0) session.cameraConfig = candidates[idx]

        val depth = session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
        val config = Config(session).apply {
            depthMode = if (depth) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
            focusMode = Config.FocusMode.AUTO
            updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
        }
        session.configure(config)
        val size = session.cameraConfig.imageSize
        return ArSetup(depth, size.width, size.height)
    }
}

/**
 * Index of the CPU image size (`[w, h]`) with the largest area whose long side is ≤ 1920;
 * if none qualifies, the smallest one; -1 for an empty list.
 */
internal fun pickCameraConfigIndex(dims: List<IntArray>, maxLongSide: Int = 1920): Int {
    if (dims.isEmpty()) return -1
    val area = { i: Int -> dims[i][0].toLong() * dims[i][1] }
    val ok = dims.indices.filter { maxOf(dims[it][0], dims[it][1]) <= maxLongSide }
    return if (ok.isNotEmpty()) ok.maxBy(area) else dims.indices.minBy(area)
}
