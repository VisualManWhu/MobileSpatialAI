package dev.recon3d.viewer

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Handler
import android.os.Looper
import android.webkit.JavascriptInterface
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.webkit.WebViewAssetLoader
import dev.recon3d.rooms.RoomRepository
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileInputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

private const val ASSET_HOST = "appassets.androidplatform.net"
private const val VIEWER_URL =
    "https://$ASSET_HOST/assets/viewer/index.html?scene=/scene/${ScenePathPolicy.SCENE_NAME}"

private const val MSG_LOADING = "正在加载场景…"
private const val MSG_NO_SCENE = "这次拍摄还没有可查看的场景"
private const val MSG_PAGE_FAILED = "查看器页面加载失败"
private const val MSG_RETRY = "重试"

/** App-side watchdog: if the page never calls back, stop spinning (viewer.js times out at 45 s). */
private const val WATCHDOG_MS = 60_000L

private sealed interface ViewerState {
    data object Loading : ViewerState
    data class Loaded(val count: Int) : ViewerState
    data class Failed(val message: String) : ViewerState
}

/** 3DGS viewer for one capture's downloaded `scene.splat` (bundled three.js + Spark in a WebView). */
@Composable
fun ViewerScreen(repo: RoomRepository, roomId: String, captureId: String) {
    val record = remember(roomId, captureId) { runCatching { repo.capture(roomId, captureId) }.getOrNull() }
    if (ScenePathPolicy.resolve(ScenePathPolicy.SCENE_NAME, record?.scenePath) == null) {
        Message(MSG_NO_SCENE)
        return
    }
    record!!
    // The camera starts at the trajectory centre. Older records have no cached HomeView: derive it
    // from the capture zip once (a few hundred JSON lines) and store it; no zip -> viewer default.
    val home by produceState<Result<HomeView?>?>(initialValue = record.home?.let { Result.success(it) }, record.id) {
        if (value != null) return@produceState
        val computed = withContext(Dispatchers.IO) { HomeView.fromCaptureZip(File(record.zipPath)) }
        if (computed != null) runCatching { repo.editCapture(roomId, record.id) { it.copy(home = computed) } }
        value = Result.success(computed)
    }
    when (val h = home) {
        null -> Message(MSG_LOADING)
        else -> SceneViewer(record.scenePath!!, home = h.getOrNull())
    }
}

/**
 * WebView viewer for one `.splat` file at [scenePath] (also used by the capture preview).
 * [home]: where the camera starts (see [HomeView]); null leaves the page's own default.
 */
@Composable
internal fun SceneViewer(scenePath: String, home: HomeView? = null) {
    // Each retry bumps the attempt: a fresh WebView, a fresh watchdog, stale callbacks ignored.
    var attempt by remember(scenePath) { mutableIntStateOf(0) }
    var state by remember(scenePath) { mutableStateOf<ViewerState>(ViewerState.Loading) }

    LaunchedEffect(scenePath, attempt) {
        delay(WATCHDOG_MS)
        // Neither onLoaded nor onError arrived (e.g. the page hung before reaching its handlers).
        if (state == ViewerState.Loading) state = ViewerState.Failed(MSG_PAGE_FAILED)
    }

    Box(Modifier.fillMaxSize().background(Color(0xFF111111))) {
        key(attempt) {
            val mine = attempt
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { context ->
                    createViewerWebView(
                        context = context,
                        scenePath = scenePath,
                        home = home,
                        onLoaded = { if (mine == attempt) state = ViewerState.Loaded(it) },
                        onError = { if (mine == attempt) state = ViewerState.Failed(it) },
                    )
                },
                onRelease = { it.destroy() },
            )
        }
        when (val s = state) {
            ViewerState.Loading -> Box(
                Modifier.fillMaxSize().background(Color(0xFF111111)),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    CircularProgressIndicator()
                    Text(MSG_LOADING, color = Color.White, style = MaterialTheme.typography.bodyLarge)
                }
            }
            is ViewerState.Failed -> Message(s.message, onRetry = {
                state = ViewerState.Loading
                attempt++
            })
            is ViewerState.Loaded -> Unit
        }
    }
}

@Composable
private fun Message(text: String, onRetry: (() -> Unit)? = null) {
    Box(Modifier.fillMaxSize().background(Color(0xFF111111)).padding(24.dp), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text(text, color = Color.White, style = MaterialTheme.typography.bodyLarge)
            if (onRetry != null) Button(onClick = onRetry) { Text(MSG_RETRY) }
        }
    }
}

/** Callbacks from viewer.js. Only the two annotated methods are visible to the page. */
private class ViewerBridge(
    private val onLoadedMain: (Int) -> Unit,
    private val onErrorMain: (String) -> Unit,
) {
    private val main = Handler(Looper.getMainLooper())

    @JavascriptInterface
    fun onLoaded(count: Int) {
        main.post { onLoadedMain(count) }
    }

    @JavascriptInterface
    fun onError(msgZh: String) {
        main.post { onErrorMain(msgZh.take(200)) }
    }
}

@SuppressLint("SetJavaScriptEnabled") // JS runs only our bundled page; other origins are blocked below.
private fun createViewerWebView(
    context: Context,
    scenePath: String,
    home: HomeView?,
    onLoaded: (Int) -> Unit,
    onError: (String) -> Unit,
): WebView {
    if (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
        WebView.setWebContentsDebuggingEnabled(true) // chrome://inspect for manual flyTo() tests
    }
    val loader = WebViewAssetLoader.Builder()
        .setDomain(ASSET_HOST)
        .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(context))
        .addPathHandler("/scene/") { path ->
            ScenePathPolicy.resolve(path, scenePath)?.let {
                WebResourceResponse("application/octet-stream", null, FileInputStream(it))
            }
        }
        .build()

    return WebView(context).apply {
        setBackgroundColor(0xFF111111.toInt())
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = false
        settings.allowFileAccess = false
        settings.allowContentAccess = false
        addJavascriptInterface(ViewerBridge(onLoaded, onError), "Android")
        webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? {
                // Never touch the network: unknown hosts get 403, unmatched paths (incl. any
                // scene path other than the one allowed) get 404.
                if (request.url.host != ASSET_HOST) return emptyResponse(403, "Forbidden")
                return loader.shouldInterceptRequest(request.url) ?: emptyResponse(404, "Not Found")
            }

            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean = true

            override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
                if (request.isForMainFrame) onError(MSG_PAGE_FAILED)
            }

            override fun onReceivedHttpError(view: WebView, request: WebResourceRequest, errorResponse: WebResourceResponse) {
                if (request.isForMainFrame) onError(MSG_PAGE_FAILED)
            }
        }
        loadUrl(VIEWER_URL + (home?.query() ?: ""))
    }
}

private fun emptyResponse(status: Int, reason: String) =
    WebResourceResponse("text/plain", "utf-8", status, reason, emptyMap(), ByteArrayInputStream(ByteArray(0)))
