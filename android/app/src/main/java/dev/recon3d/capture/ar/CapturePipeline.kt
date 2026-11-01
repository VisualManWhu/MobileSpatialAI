package dev.recon3d.capture.ar

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import dev.recon3d.capture.CapturePackageWriter
import dev.recon3d.capture.ImageConvert
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

/** Raw bytes copied out of an ARCore frame on the GL thread; converted and written on the IO worker. */
class FrameJob(
    val tNs: Long,
    val width: Int, val height: Int,
    val y: ByteArray, val yRowStride: Int, val yPixelStride: Int,
    val u: ByteArray, val v: ByteArray, val uvRowStride: Int, val uvPixelStride: Int,
    val depth: DepthBytes?,
    val pose: FloatArray,
    val fx: Float, val fy: Float, val cx: Float, val cy: Float,
)

/** [intrinsics]: optional `[fx, fy, cx, cy]` in depth-image pixels (see [DepthIntrinsics]). */
class DepthBytes(
    val data: ByteArray, val width: Int, val height: Int, val rowStrideBytes: Int,
    val intrinsics: FloatArray? = null,
)

/** Depth-image intrinsics: GPU-texture intrinsics scaled to the depth resolution. */
object DepthIntrinsics {
    fun scaled(focal: FloatArray, principal: FloatArray, dims: IntArray, depthW: Int, depthH: Int): FloatArray {
        val sx = depthW.toFloat() / dims[0]
        val sy = depthH.toFloat() / dims[1]
        return floatArrayOf(focal[0] * sx, focal[1] * sy, principal[0] * sx, principal[1] * sy)
    }
}

/**
 * Bounded keyframe queue (max [MAX_PENDING] jobs) consumed by ONE IO coroutine that does YUV→JPEG and
 * [CapturePackageWriter.addFrame]. The writer is created lazily on the first job with depth size
 * [depthSize] (null → no depth for this capture). A job that throws is logged, counted in [failed] and
 * skipped; the capture continues.
 */
class CapturePipeline(
    private val stagingDir: File,
    private val roomId: String,
    private val captureId: String,
    private val device: String,
    private val depthSize: () -> IntArray?,
    private val encodeJpeg: (FrameJob) -> ByteArray = ::yuvJobToJpeg,
    private val log: (String, Throwable?) -> Unit = { m, e -> Log.w(TAG, m, e) },
) {
    private val queue = Channel<FrameJob>(Channel.UNLIMITED)
    private val pending = AtomicInteger(0)
    private val _kept = MutableStateFlow(0)
    private val _dropped = MutableStateFlow(0)
    private val _failed = MutableStateFlow(0)

    /** Frames written so far. */
    val kept: StateFlow<Int> = _kept
    /** Keyframes dropped because the queue was full. */
    val dropped: StateFlow<Int> = _dropped
    /** Keyframes whose conversion / write threw (skipped). */
    val failed: StateFlow<Int> = _failed

    @OptIn(ExperimentalCoroutinesApi::class)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO.limitedParallelism(1))

    private val result = scope.async {
        var writer: CapturePackageWriter? = null
        var depthW: Int? = null
        var depthH: Int? = null
        for (job in queue) {
            try {
                val w = writer ?: run {
                    val d = depthSize()
                    depthW = d?.get(0); depthH = d?.get(1)
                    stagingDir.mkdirs()
                    CapturePackageWriter(
                        dir = stagingDir, roomId = roomId, captureId = captureId,
                        imageW = job.width, imageH = job.height, depthW = depthW, depthH = depthH, device = device,
                    ).also { writer = it }
                }
                val jpeg = encodeJpeg(job)
                val d = job.depth?.takeIf { it.width == depthW && it.height == depthH }
                val depth = d?.let { ImageConvert.depth16ToMm(it.data, it.width, it.height, it.rowStrideBytes) }
                w.addFrame(job.tNs, jpeg, depth, job.pose, job.fx, job.fy, job.cx, job.cy, depthIntrinsics = d?.intrinsics)
                _kept.value++
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _failed.value++
                log("keyframe t=${job.tNs} failed, skipped (failed=${_failed.value})", e)
            } finally {
                pending.decrementAndGet()
            }
        }
        writer?.finish()
    }

    /** True if there is room for one more job; call before acquiring images to avoid wasted copies. */
    fun hasRoom(): Boolean {
        if (pending.get() < MAX_PENDING) return true
        _dropped.value++
        log("keyframe dropped, queue full (dropped=${_dropped.value})", null)
        return false
    }

    /** Enqueues a job (GL thread). Silently ignored after [finish]. */
    fun submit(job: FrameJob) {
        pending.incrementAndGet()
        if (queue.trySend(job).isFailure) pending.decrementAndGet()
    }

    /** True once at least one keyframe was queued or written. */
    fun hasFrames(): Boolean = _kept.value > 0 || pending.get() > 0

    /** Stops accepting jobs, drains the queue and zips the package. Returns null if no frame was written. */
    suspend fun finish(): File? {
        queue.close()
        return result.await()
    }

    /** Discards the capture (screen left without finishing): drains, then deletes the zip and staging dir. */
    fun abandon() {
        queue.close()
        scope.launch {
            runCatching { result.await() }.getOrNull()?.delete()
            stagingDir.deleteRecursively()
        }
    }

    private companion object {
        const val TAG = "CapturePipeline"
        const val MAX_PENDING = 8
    }
}

private const val JPEG_QUALITY = 90

/** YUV_420_888 bytes → NV21 → JPEG (quality 90). Android-only (YuvImage). */
fun yuvJobToJpeg(job: FrameJob): ByteArray {
    val nv21 = ImageConvert.yuv420ToNv21(
        job.width, job.height, job.y, job.yRowStride, job.yPixelStride,
        job.u, job.v, job.uvRowStride, job.uvPixelStride,
    )
    val out = ByteArrayOutputStream(job.width * job.height / 4)
    YuvImage(nv21, ImageFormat.NV21, job.width, job.height, null)
        .compressToJpeg(Rect(0, 0, job.width, job.height), JPEG_QUALITY, out)
    return out.toByteArray()
}
