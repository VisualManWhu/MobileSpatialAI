package dev.recon3d.capture.ar

import android.content.res.AssetManager
import android.media.Image
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.util.Log
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.SessionPausedException
import dev.recon3d.capture.CaptureGuide
import dev.recon3d.capture.GuideUi
import dev.recon3d.capture.KeyframeSelector
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * GL-thread frame loop: session.update → camera feed → [CaptureGuide] → (when recording) [KeyframeSelector]
 * → copy image bytes into a [FrameJob] for [CapturePipeline]. [guide] and [selector] are only touched on
 * the GL thread (UI actions go through `GLSurfaceView.queueEvent`).
 */
class CaptureRenderer(
    private val assets: AssetManager,
    private val session: Session,
    private val pipeline: CapturePipeline,
    private val depthEnabled: Boolean,
    private val displayRotation: () -> Int,
) : GLSurfaceView.Renderer {
    val guide = CaptureGuide()
    private val selector = KeyframeSelector()
    private val background = BackgroundRenderer()
    private val pose = FloatArray(16)
    private var lastTimestamp = 0L
    private var viewW = 0
    private var viewH = 0
    private var geometryDirty = false
    private var depthProbeCountdown = 0

    private val _ui = MutableStateFlow(guide.last)
    val ui: StateFlow<GuideUi> = _ui

    /** Set from the UI thread; frames are only kept while true. */
    @Volatile var recording = false

    /** GL thread only (use `queueEvent`). */
    fun skipStage() {
        guide.skipStage()
        _ui.value = guide.last
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0f, 0f, 0f, 1f)
        background.createOnGlThread(assets)
        session.setCameraTextureName(background.cameraTextureId)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
        viewW = width; viewH = height; geometryDirty = true
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        if (geometryDirty) {
            session.setDisplayGeometry(displayRotation(), viewW, viewH)
            geometryDirty = false
        }
        val frame = try {
            session.update()
        } catch (e: SessionPausedException) {
            return
        } catch (e: Exception) {
            Log.w(TAG, "session.update failed", e); return
        }
        background.updateDisplayGeometry(frame)
        val t = frame.timestamp
        if (t == 0L) return // camera has not produced an image yet
        background.draw()
        if (t == lastTimestamp) return // LATEST_CAMERA_IMAGE may repeat a frame
        lastTimestamp = t

        val camera = frame.camera
        val tracking = camera.trackingState == TrackingState.TRACKING
        camera.pose.toMatrix(pose, 0) // world_from_camera, column-major, OpenGL convention
        _ui.value = guide.onFrame(t, pose, tracking)

        if (!recording) {
            probeDepthSize(frame)
            return
        }
        if (selector.shouldKeep(t, pose, tracking) && pipeline.hasRoom()) {
            buildJob(frame, camera, t)?.let(pipeline::submit)
        }
    }

    /** Before recording starts, learn the depth image size so the writer can be created with it. */
    private fun probeDepthSize(frame: Frame) {
        if (!depthEnabled || pipeline.knownDepthSize != null || --depthProbeCountdown > 0) return
        depthProbeCountdown = 30
        try {
            frame.acquireDepthImage16Bits().use { pipeline.knownDepthSize = intArrayOf(it.width, it.height) }
        } catch (_: NotYetAvailableException) {
        }
    }

    private fun buildJob(frame: Frame, camera: Camera, t: Long): FrameJob? {
        val image = try {
            frame.acquireCameraImage()
        } catch (_: NotYetAvailableException) {
            return null
        }
        val job = image.use { img ->
            val (yP, uP, vP) = img.planes
            val depth = if (depthEnabled) acquireDepth(frame) else null
            // Intrinsics are for imageIntrinsics.imageDimensions; scale to the actual CPU image (normally equal).
            val intr = camera.imageIntrinsics
            val dims = intr.imageDimensions
            val sx = img.width.toFloat() / dims[0]
            val sy = img.height.toFloat() / dims[1]
            val f = intr.focalLength
            val c = intr.principalPoint
            FrameJob(
                tNs = t, width = img.width, height = img.height,
                y = yP.copyBytes(), yRowStride = yP.rowStride, yPixelStride = yP.pixelStride,
                u = uP.copyBytes(), v = vP.copyBytes(), uvRowStride = uP.rowStride, uvPixelStride = uP.pixelStride,
                depth = depth, pose = pose.copyOf(),
                fx = f[0] * sx, fy = f[1] * sy, cx = c[0] * sx, cy = c[1] * sy,
            )
        }
        return job
    }

    private fun acquireDepth(frame: Frame): DepthBytes? = try {
        frame.acquireDepthImage16Bits().use { d ->
            val p = d.planes[0]
            DepthBytes(p.copyBytes(), d.width, d.height, p.rowStride).also {
                if (pipeline.knownDepthSize == null) pipeline.knownDepthSize = intArrayOf(d.width, d.height)
            }
        }
    } catch (_: NotYetAvailableException) {
        null
    }

    private fun Image.Plane.copyBytes(): ByteArray {
        val b = buffer.duplicate()
        b.rewind()
        return ByteArray(b.remaining()).also { b.get(it) }
    }

    private companion object {
        const val TAG = "CaptureRenderer"
    }
}
