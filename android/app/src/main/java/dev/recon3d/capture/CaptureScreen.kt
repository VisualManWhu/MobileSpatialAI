package dev.recon3d.capture

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import dev.recon3d.capture.ar.ArSessionHelper
import dev.recon3d.capture.ar.CapturePipeline
import dev.recon3d.capture.ar.CaptureRenderer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

private val Blue = Color(0xFF2F80ED)
private val Yellow = Color(0xFFFFD600)
private val RecordRed = Color(0xFFE53935)
private val Scrim = Color.Black.copy(alpha = 0.45f)

/** What a finished capture produced: the package zip, the preview point cloud (null if empty) and the keyframe count. */
class CaptureOutput(val zip: File, val points: File?, val frames: Int, val pointCount: Int)

/**
 * AR capture screen: live camera feed, a red record button and the keyframe count; warnings when ARCore
 * loses tracking or the phone turns too fast. "完成拍摄" drains the queue, writes
 * `filesDir/captures/<roomId>/<captureId>.zip` plus `<captureId>.points.splat`, then calls [onFinished].
 */
@Composable
fun CaptureScreen(roomId: String, onFinished: (CaptureOutput) -> Unit, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val activity = LocalActivity.current ?: return
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    // Permanently denied = a request was answered "deny" and the system will no longer show the dialog
    // (shouldShowRequestPermissionRationale is false after a denial).
    var permanentlyDenied by remember { mutableStateOf(false) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        granted = ok
        permanentlyDenied = !ok && !activity.shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)
    }
    // Coming back from system settings: re-check the permission.
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val obs = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME && !granted) {
                granted = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
            }
        }
        lifecycleOwner.lifecycle.addObserver(obs)
        onDispose { lifecycleOwner.lifecycle.removeObserver(obs) }
    }

    // Portrait only: a rotation would recreate the GL surface mid-capture.
    DisposableEffect(activity) {
        val old = activity.requestedOrientation
        activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        onDispose { activity.requestedOrientation = old }
    }

    Box(modifier.fillMaxSize().background(Color.Black)) {
        if (granted) {
            ArCapture(activity, roomId, onFinished)
        } else {
            Column(
                Modifier.align(Alignment.Center).padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                if (permanentlyDenied) {
                    Text(
                        "相机权限已被拒绝，请在系统设置中开启",
                        color = Color.White, style = MaterialTheme.typography.titleMedium, textAlign = TextAlign.Center,
                    )
                    Button(onClick = {
                        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        context.startActivity(intent)
                    }) { Text("打开设置") }
                } else {
                    Text("拍摄需要使用相机", color = Color.White, style = MaterialTheme.typography.titleMedium)
                    Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) { Text("授予相机权限") }
                }
            }
        }
    }
}

/** Owns the ARCore session, GL view, renderer and pipeline; driven by lifecycle resume/pause. */
private class ArHost(private val activity: Activity, private val roomId: String) {
    var error by mutableStateOf<String?>(null)
    var renderer by mutableStateOf<CaptureRenderer?>(null)
        private set
    var pipeline by mutableStateOf<CapturePipeline?>(null)
        private set
    var glView: GLSurfaceView? = null
    private var session: Session? = null
    private var installRequested = false
    private var resumed = false
    private var finished = false

    fun onResume() {
        if (session == null && !createSession()) return
        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            error = "相机不可用，请关闭其他正在使用相机的应用后重试"
            return
        }
        resumed = true
        error = null // recover from a transient CameraNotAvailableException
        glView?.onResume()
    }

    fun onPause() {
        if (!resumed) return
        resumed = false
        glView?.onPause() // blocks until the GL thread stops touching the session
        session?.pause()
    }

    fun close() {
        onPause()
        glView?.onPause()
        session?.close()
        session = null
        if (!finished) pipeline?.abandon()
    }

    /**
     * Finishes recording. [FinishResult.NoFrames] leaves the capture running; on failure the broken pipeline
     * is discarded and replaced by a fresh one, so the user can record again.
     */
    suspend fun finish(): FinishResult {
        val p = pipeline ?: return FinishResult.NoFrames
        val r = renderer ?: return FinishResult.NoFrames
        if (!p.hasFrames()) return FinishResult.NoFrames
        r.recording = false
        val zip = try {
            p.finish()
        } catch (e: Exception) {
            Log.e(TAG, "finish failed", e)
            null
        }
        if (zip != null) {
            finished = true
            val snap = p.cloud.snapshot()
            val points = if (snap.size == 0) null else withContext(Dispatchers.IO) {
                val f = File(zip.parentFile, zip.nameWithoutExtension + POINTS_SUFFIX)
                runCatching { f.writeBytes(PointCloudSplat.encode(snap)) }
                    .onFailure { Log.w(TAG, "point cloud not written", it); f.delete() } // never keep a truncated file
                f.takeIf { it.isFile }
            }
            return FinishResult.Saved(CaptureOutput(zip, points, p.kept.value, if (points == null) 0 else snap.size))
        }
        p.abandon()
        val fresh = newPipeline(r)
        pipeline = fresh
        r.pipeline = fresh
        return FinishResult.Failed
    }

    private fun newPipeline(r: CaptureRenderer): CapturePipeline {
        val captureId = "cap-" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS"))
        val staging = File(activity.filesDir, "captures/$roomId/$captureId")
        return CapturePipeline(staging, roomId, captureId, Build.MODEL, depthSize = { r.depthSize })
    }

    private fun createSession(): Boolean {
        try {
            when (ArCoreApk.getInstance().requestInstall(activity, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> { installRequested = true; return false }
                ArCoreApk.InstallStatus.INSTALLED -> {}
            }
            val s = Session(activity)
            val setup = ArSessionHelper.configure(s)
            Log.i(TAG, "AR ready: image ${setup.imageW}x${setup.imageH}, depth=${setup.depthEnabled}")
            val r = CaptureRenderer(activity.assets, s, setup.depthEnabled) { activity.display?.rotation ?: 0 }
            val p = newPipeline(r)
            r.pipeline = p
            session = s
            pipeline = p
            renderer = r
            return true
        } catch (e: UnavailableException) {
            Log.e(TAG, "ARCore unavailable", e)
            error = arcoreErrorZh(e)
        } catch (e: Exception) {
            Log.e(TAG, "AR session failed", e)
            error = MSG_AR_START_FAILED
        }
        return false
    }

    private companion object {
        const val TAG = "CaptureScreen"
    }
}

/** Suffix of the preview point cloud written next to the capture zip. */
const val POINTS_SUFFIX = ".points.splat"

private sealed interface FinishResult {
    data class Saved(val output: CaptureOutput) : FinishResult
    data object NoFrames : FinishResult
    data object Failed : FinishResult
}

private fun arcoreErrorZh(e: UnavailableException): String = when (e) {
    is UnavailableArcoreNotInstalledException, is UnavailableUserDeclinedInstallationException -> "需要安装 Google Play 服务（AR）才能拍摄"
    is UnavailableApkTooOldException -> "请在应用商店更新 Google Play 服务（AR）"
    is UnavailableSdkTooOldException -> "本应用版本过旧，请更新"
    is UnavailableDeviceNotCompatibleException -> "这台手机不支持 ARCore"
    else -> MSG_AR_START_FAILED
}

private const val MSG_AR_START_FAILED = "AR 启动失败，请检查是否已安装/更新 ARCore（Google Play 服务 - AR）后重试"
internal const val MSG_NO_DEPTH_FALLBACK = "未获取到深度，本次拍摄将不含深度（重建质量可能下降）"

@Composable
private fun ArCapture(activity: Activity, roomId: String, onFinished: (CaptureOutput) -> Unit) {
    val host = remember(roomId) { ArHost(activity, roomId) }
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    DisposableEffect(lifecycle, host) {
        val obs = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> host.onResume()
                Lifecycle.Event.ON_PAUSE -> host.onPause()
                else -> {}
            }
        }
        lifecycle.addObserver(obs)
        onDispose {
            lifecycle.removeObserver(obs)
            host.close()
        }
    }

    val error = host.error
    val renderer = host.renderer
    if (error != null) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(error, color = Color.White, textAlign = TextAlign.Center, modifier = Modifier.padding(24.dp))
        }
        return
    }
    if (renderer == null) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("正在启动 AR…", color = Color.White)
        }
        return
    }

    val pipeline = host.pipeline ?: return
    val status by renderer.status.collectAsState()
    val depthReady by renderer.depthReady.collectAsState()
    val depthFallback by renderer.depthFallback.collectAsState()
    var recording by remember { mutableStateOf(false) }
    val warning = when {
        status.warning != null -> status.warning
        // R27: tell the user the capture has no depth.
        depthFallback && recording -> MSG_NO_DEPTH_FALLBACK
        else -> null
    }
    val kept by pipeline.kept.collectAsState()
    val dropped by pipeline.dropped.collectAsState()
    val failed by pipeline.failed.collectAsState()
    var finishing by remember { mutableStateOf(false) }
    var message by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    Box(Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                GLSurfaceView(ctx).apply {
                    preserveEGLContextOnPause = true
                    setEGLContextClientVersion(3)
                    setEGLConfigChooser(8, 8, 8, 8, 16, 0)
                    setRenderer(renderer)
                    renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
                    setWillNotDraw(false)
                    keepScreenOn = true
                    host.glView = this
                }
            },
        )
        CaptureOverlay(
            warning = warning,
            depthReady = depthReady,
            recording = recording,
            kept = kept,
            dropped = dropped,
            failed = failed,
            message = message,
            onRecord = {
                if (!recording && !depthReady) {
                    message = "正在等待深度传感器，请稍候"
                    return@CaptureOverlay
                }
                recording = !recording
                renderer.recording = recording
                message = null
            },
            onFinish = {
                if (finishing) return@CaptureOverlay
                finishing = true
                scope.launch {
                    try {
                        when (val r = host.finish()) {
                            is FinishResult.Saved -> onFinished(r.output)
                            FinishResult.NoFrames -> message = "还没有拍到画面，请先点红色按钮开始拍摄"
                            FinishResult.Failed -> {
                                recording = false
                                message = "保存失败，请返回重拍（已重置，可重新点红色按钮拍摄）"
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("CaptureScreen", "finish failed", e)
                        message = "保存失败，请返回重拍"
                    } finally {
                        finishing = false
                    }
                }
            },
        )
        if (finishing) {
            Box(Modifier.fillMaxSize().background(Scrim), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    CircularProgressIndicator(color = Color.White)
                    Text("正在保存拍摄…", color = Color.White)
                }
            }
        }
    }
}

@Composable
private fun CaptureOverlay(
    warning: String?,
    depthReady: Boolean,
    recording: Boolean,
    kept: Int,
    dropped: Int,
    failed: Int,
    message: String?,
    onRecord: () -> Unit,
    onFinish: () -> Unit,
) {
    Box(Modifier.fillMaxSize()) {
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().background(Scrim).padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            warning?.let { w ->
                Text(
                    "⚠ $w",
                    color = Color.Black,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(Yellow).padding(10.dp),
                )
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.weight(1f))
                RecordButton(recording, onRecord)
                Box(Modifier.weight(1f), contentAlignment = Alignment.Center) {
                    if (kept > 0) {
                        Button(onClick = onFinish, colors = ButtonDefaults.buttonColors(containerColor = Blue)) { Text("完成拍摄") }
                    }
                }
            }
            val status = when {
                message != null -> message
                !depthReady -> "正在等待深度传感器…"
                recording -> "录制中 · 已拍 $kept 张" + lossNote(dropped, failed)
                kept > 0 -> "已暂停 · 已拍 $kept 张"
                else -> "点红色按钮开始拍摄，慢慢绕房间走一圈"
            }
            Text(status, color = if (message != null) Yellow else Color.White.copy(alpha = 0.85f), fontSize = 14.sp)
        }
    }
}

@Composable
private fun RecordButton(recording: Boolean, onClick: () -> Unit) {
    Box(
        Modifier.size(76.dp).clip(CircleShape).border(4.dp, Color.White, CircleShape).clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        if (recording) {
            Box(Modifier.size(30.dp).clip(RoundedCornerShape(6.dp)).background(RecordRed))
        } else {
            Box(Modifier.size(58.dp).clip(CircleShape).background(RecordRed))
        }
    }
}

private fun lossNote(dropped: Int, failed: Int): String {
    val parts = listOfNotNull(
        "丢弃 $dropped".takeIf { dropped > 0 },
        "失败 $failed".takeIf { failed > 0 },
    )
    return if (parts.isEmpty()) "" else parts.joinToString("，", "（", "）")
}
