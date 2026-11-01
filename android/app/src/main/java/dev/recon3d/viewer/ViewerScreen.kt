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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
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
import java.io.FileInputStream

private const val ASSET_HOST = "appassets.androidplatform.net"
private const val VIEWER_URL =
    "https://$ASSET_HOST/assets/viewer/index.html?scene=/scene/${ScenePathPolicy.SCENE_NAME}"

private const val MSG_LOADING = "正在加载场景…"
private const val MSG_NO_SCENE = "这次拍摄还没有可查看的场景"
private const val MSG_PAGE_FAILED = "查看器页面加载失败"

private sealed interface ViewerState {
    data object Loading : ViewerState
    data class Loaded(val count: Int) : ViewerState
    data class Failed(val message: String) : ViewerState
}

/** 3DGS viewer for one capture's downloaded `scene.splat` (bundled three.js + Spark in a WebView). */
@Composable
fun ViewerScreen(repo: RoomRepository, roomId: String, captureId: String) {
    val scenePath = remember(roomId, captureId) {
        runCatching { repo.capture(roomId, captureId).scenePath }.getOrNull()
    }
    if (ScenePathPolicy.resolve(ScenePathPolicy.SCENE_NAME, scenePath) == null) {
        Message(MSG_NO_SCENE)
        return
    }
    SceneViewer(scenePath!!)
}

@Composable
private fun SceneViewer(scenePath: String) {
    var state by remember(scenePath) { mutableStateOf<ViewerState>(ViewerState.Loading) }

    Box(Modifier.fillMaxSize().background(Color(0xFF111111))) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { context ->
                createViewerWebView(
                    context = context,
                    scenePath = scenePath,
                    onLoaded = { state = ViewerState.Loaded(it) },
                    onError = { state = ViewerState.Failed(it) },
                )
            },
            onRelease = { it.destroy() },
        )
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
            is ViewerState.Failed -> Message(s.message)
            is ViewerState.Loaded -> Unit
        }
    }
}

@Composable
private fun Message(text: String) {
    Box(Modifier.fillMaxSize().background(Color(0xFF111111)).padding(24.dp), contentAlignment = Alignment.Center) {
        Text(text, color = Color.White, style = MaterialTheme.typography.bodyLarge)
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
        loadUrl(VIEWER_URL)
    }
}

private fun emptyResponse(status: Int, reason: String) =
    WebResourceResponse("text/plain", "utf-8", status, reason, emptyMap(), ByteArrayInputStream(ByteArray(0)))
