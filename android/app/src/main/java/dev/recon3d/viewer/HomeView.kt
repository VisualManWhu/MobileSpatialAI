package dev.recon3d.viewer

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.util.Locale
import java.util.zip.ZipFile
import kotlin.math.hypot

/**
 * Where the viewer's camera starts for one capture, in the scene world frame (ARCore world, +Y up,
 * same as the `.splat` positions): [x],[y],[z] is the centre of the capture trajectory (mean camera
 * position), [lx],[ly],[lz] the first frame's view direction. The viewer levels the pitch itself.
 */
@Serializable
data class HomeView(val x: Float, val y: Float, val z: Float, val lx: Float, val ly: Float, val lz: Float) {
    /** Query-string suffix for the viewer page (`&home=x,y,z&look=dx,dy,dz`). */
    fun query(): String = "&home=${f(x)},${f(y)},${f(z)}&look=${f(lx)},${f(ly)},${f(lz)}"

    private fun f(v: Float) = String.format(Locale.ROOT, "%.4f", v)

    companion object {
        /**
         * poses: 16 floats each, column-major world_from_camera (OpenGL: camera looks down -Z), as
         * written to `frames.jsonl`. Null for no poses or non-finite values.
         */
        fun fromPoses(poses: List<FloatArray>): HomeView? {
            if (poses.isEmpty()) return null
            var sx = 0.0; var sy = 0.0; var sz = 0.0
            for (p in poses) {
                require(p.size == 16) { "pose must have 16 floats" }
                sx += p[12]; sy += p[13]; sz += p[14]
            }
            val n = poses.size.toDouble()
            val first = poses.first()
            // Forward = -(third column); fall back to -Z when the first frame looks (nearly) straight up/down.
            var lx = -first[8]; var ly = -first[9]; var lz = -first[10]
            if (hypot(lx, lz) < 1e-4f) { lx = 0f; ly = 0f; lz = -1f }
            val h = HomeView((sx / n).toFloat(), (sy / n).toFloat(), (sz / n).toFloat(), lx, ly, lz)
            return h.takeIf { listOf(it.x, it.y, it.z, it.lx, it.ly, it.lz).all(Float::isFinite) }
        }

        /** Reads `frames.jsonl` from a capture package; null when the zip or the poses are unusable. */
        fun fromCaptureZip(zip: File): HomeView? = runCatching {
            ZipFile(zip).use { z ->
                val entry = z.getEntry("frames.jsonl") ?: return null
                val poses = z.getInputStream(entry).bufferedReader().useLines { lines ->
                    lines.filter { it.isNotBlank() }
                        .map { line -> Json.parseToJsonElement(line).jsonObject["pose"]!!.jsonArray.map { it.jsonPrimitive.float }.toFloatArray() }
                        .toList()
                }
                fromPoses(poses)
            }
        }.getOrNull()
    }
}
