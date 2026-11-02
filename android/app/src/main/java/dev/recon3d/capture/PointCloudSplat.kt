package dev.recon3d.capture

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Writes a [PointCloudSnapshot] as an antimatter15 `.splat` (the viewer's scene format): per point
 * pos f32x3, scale f32x3, rgba u8x4, rot u8x4 (wxyz, `q * 128 + 128`). Every point becomes a small,
 * fully opaque, axis-aligned Gaussian in its own colour, so the existing viewer can show it unchanged.
 */
object PointCloudSplat {
    const val STRIDE = 32
    /** Radius of each point as a Gaussian, metres (a bit over half the 2 cm voxel, so the cloud reads as surfaces). */
    const val POINT_SCALE_M = 0.012f

    fun encode(snap: PointCloudSnapshot): ByteArray {
        val buf = ByteBuffer.allocate(snap.size * STRIDE).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until snap.size) {
            buf.putFloat(snap.xyz[3 * i]); buf.putFloat(snap.xyz[3 * i + 1]); buf.putFloat(snap.xyz[3 * i + 2])
            buf.putFloat(POINT_SCALE_M); buf.putFloat(POINT_SCALE_M); buf.putFloat(POINT_SCALE_M)
            buf.put(snap.rgb[3 * i]); buf.put(snap.rgb[3 * i + 1]); buf.put(snap.rgb[3 * i + 2]); buf.put(0xFF.toByte())
            // identity rotation: w = 1 -> 255 (clipped), x = y = z = 0 -> 128
            buf.put(0xFF.toByte()); buf.put(0x80.toByte()); buf.put(0x80.toByte()); buf.put(0x80.toByte())
        }
        return buf.array()
    }
}
