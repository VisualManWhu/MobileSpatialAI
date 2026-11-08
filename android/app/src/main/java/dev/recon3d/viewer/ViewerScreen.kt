package dev.recon3d.viewer

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.ActivityInfo
import android.content.pm.ApplicationInfo
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
import android.webkit.JavascriptInterface
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.compose.LocalActivity
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
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
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.webkit.WebViewAssetLoader
import dev.recon3d.rooms.CaptureRecord
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
private const val MENU_RERUN_OBJECTS = "重新识别"
private const val CLEAR_HIGHLIGHT = "取消高亮"

/** App-side watchdog: if the page never calls back, stop spinning (viewer.js times out at 45 s). */
private const val WATCHDOG_MS = 60_000L

private sealed interface ViewerState {
    data object Loading : ViewerState
    data class Loaded(val count: Int) : ViewerState
    data class Failed(val message: String) : ViewerState
}

/**
 * 3DGS viewer for one capture's downloaded `scene.splat` (bundled three.js + Spark in a WebView),
 * plus the phase-2 objects chip / list / highlight driven by [objects].
 */
@Composable
fun ViewerScreen(repo: RoomRepository, roomId: String, captureId: String, objects: ObjectsViewModel) {
    // Observed live: the objects chip follows the recognition progress written by ObjectsViewModel.
    val rooms by repo.rooms().collectAsState()
    val record = rooms.firstOrNull { it.id == roomId }?.captures?.firstOrNull { it.id == captureId }
    if (record == null || ScenePathPolicy.resolve(ScenePathPolicy.SCENE_NAME, record.scenePath) == null) {
        Message(MSG_NO_SCENE)
        return
    }
    LaunchedEffect(record.id) { objects.resume(roomId, record.id) }
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
        else -> SceneViewer(record.scenePath!!, home = h.getOrNull(), objectsPath = record.objectsPath, objectsVersion = record.objectsVersion) { webView, loaded ->
            ObjectsOverlay(record = record, loaded = loaded, webView = webView, onStart = { objects.start(roomId, record.id) })
        }
    }
}

/**
 * WebView viewer for one `.splat` file at [scenePath] (also used by the capture preview).
 * [home]: where the camera starts (see [HomeView]); null leaves the page's own default.
 * [objectsPath]: directory of the unpacked object table, served to the page as `/scene/objects.bin`.
 * [overlay]: extra Compose content over the page, given the live WebView and whether the scene loaded.
 */
@Composable
internal fun SceneViewer(
    scenePath: String,
    home: HomeView? = null,
    objectsPath: String? = null,
    objectsVersion: Int = 0,
    overlay: @Composable (webView: WebView?, loaded: Boolean) -> Unit = { _, _ -> },
) {
    // Each retry bumps the attempt: a fresh WebView, a fresh watchdog, stale callbacks ignored.
    var attempt by remember(scenePath) { mutableIntStateOf(0) }
    var state by remember(scenePath) { mutableStateOf<ViewerState>(ViewerState.Loading) }
    var roaming by remember(scenePath) { mutableStateOf(false) }
    var webView by remember(scenePath) { mutableStateOf<WebView?>(null) }
    // The asset loader reads this on every /scene/objects.bin request: a new table needs no new WebView.
    val objectsPathHolder = remember(scenePath) { arrayOfNulls<String>(1) }
    objectsPathHolder[0] = objectsPath

    LaunchedEffect(scenePath, attempt) {
        delay(WATCHDOG_MS)
        // Neither onLoaded nor onError arrived (e.g. the page hung before reaching its handlers).
        if (state == ViewerState.Loading) state = ViewerState.Failed(MSG_PAGE_FAILED)
    }

    // 漫游 mode is landscape-only (two thumbs on the joysticks); orbit mode follows the sensor.
    // The manifest's configChanges keeps the WebView alive across the rotation.
    val activity = LocalActivity.current
    DisposableEffect(activity, roaming) {
        if (activity == null) return@DisposableEffect onDispose {}
        val lock = roaming // snapshot: by dispose time the state has already flipped
        val old = activity.requestedOrientation
        if (lock) activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        onDispose { if (lock) activity.requestedOrientation = old }
    }

    val loaded = state is ViewerState.Loaded
    // Whenever the scene is up and an object table exists (or a new one was unpacked), hand the ids to the page.
    LaunchedEffect(loaded, objectsPath, objectsVersion, webView) {
        val wv = webView ?: return@LaunchedEffect
        if (loaded && objectsPath != null) wv.evaluateJavascript("window.loadObjects('/scene/${ScenePathPolicy.OBJECTS_NAME}')", null)
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
                        objectsPath = { objectsPathHolder[0] },
                        home = home,
                        onLoaded = { if (mine == attempt) state = ViewerState.Loaded(it) },
                        onError = { if (mine == attempt) state = ViewerState.Failed(it) },
                        onRoam = { if (mine == attempt) roaming = it },
                    ).also { webView = it }
                },
                onRelease = {
                    roaming = false // a fresh WebView (retry) starts in orbit mode
                    if (webView === it) webView = null
                    it.destroy()
                },
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
        overlay(webView, loaded)
    }
}

/** Objects chip (top-right), the object list sheet and the highlight bar. */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
private fun ObjectsOverlay(record: CaptureRecord, loaded: Boolean, webView: WebView?, onStart: () -> Unit) {
    var sheetOpen by remember { mutableStateOf(false) }
    var menuOpen by remember { mutableStateOf(false) }
    var highlighted by remember(record.id) { mutableStateOf<Pair<Int, String>?>(null) } // id to title
    val snackbar = remember { SnackbarHostState() }
    val table by produceState<ObjectsTable?>(initialValue = null, record.objectsPath, record.objectsVersion) {
        value = record.objectsPath?.let { dir -> withContext(Dispatchers.IO) { runCatching { ObjectsTable.load(File(dir)) }.getOrNull() } }
    }
    LaunchedEffect(record.objectsState, record.objectsMessage) {
        objectsFailureText(record)?.let { snackbar.showSnackbar(it) }
    }
    val hasTable = table != null
    val action = objectsChipAction(record, hasTable)
    val offersRerun = objectsMenuOffersRerun(record, hasTable)
    // A new table replaces the list and the page's ids: any highlight now points at stale ids.
    LaunchedEffect(record.objectsVersion) { highlighted = null }

    Box(Modifier.fillMaxSize()) {
        if (loaded) {
            Box(Modifier.align(Alignment.TopEnd).padding(12.dp)) {
                Surface(
                    shape = RoundedCornerShape(999.dp),
                    color = Color.White.copy(alpha = 0.18f),
                    contentColor = Color.White,
                    modifier = Modifier.combinedClickable(
                        onClick = {
                            when (action) {
                                ChipAction.OPEN_LIST -> sheetOpen = true
                                ChipAction.START -> onStart()
                                ChipAction.NONE -> Unit
                            }
                        },
                        onLongClick = { if (offersRerun) menuOpen = true },
                    ),
                ) {
                    Text(objectsChipLabel(record), Modifier.padding(horizontal = 14.dp, vertical = 6.dp), style = MaterialTheme.typography.bodyMedium)
                }
                DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                    DropdownMenuItem(text = { Text(MENU_RERUN_OBJECTS) }, onClick = { menuOpen = false; onStart() })
                }
            }
        }
        highlighted?.let { (_, title) ->
            Surface(
                shape = RoundedCornerShape(999.dp),
                color = Color.White.copy(alpha = 0.18f),
                contentColor = Color.White,
                modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 20.dp).clickable {
                    webView?.evaluateJavascript("window.highlightObject(0)", null)
                    highlighted = null
                },
            ) {
                Text("$title · $CLEAR_HIGHLIGHT", Modifier.padding(horizontal = 16.dp, vertical = 8.dp))
            }
        }
        SnackbarHost(snackbar, Modifier.align(Alignment.BottomCenter))
    }

    val t = table
    if (sheetOpen && t != null) {
        ModalBottomSheet(onDismissRequest = { sheetOpen = false }) {
            val dir = record.objectsPath
            LazyColumn(Modifier.fillMaxWidth().padding(bottom = 24.dp)) {
                for (group in t.grouped()) {
                    stickyHeader {
                        Surface(color = MaterialTheme.colorScheme.surface, modifier = Modifier.fillMaxWidth()) {
                            Text(groupTitle(group), Modifier.padding(horizontal = 16.dp, vertical = 8.dp), style = MaterialTheme.typography.titleSmall)
                        }
                    }
                    items(group.items.size) { i ->
                        val entry = group.items[i]
                        val title = itemTitle(group, i)
                        Row(
                            Modifier.fillMaxWidth().clickable {
                                webView?.evaluateJavascript(flyToObjectJs(entry), null)
                                highlighted = entry.id to title
                                sheetOpen = false
                            }.padding(horizontal = 16.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            Thumbnail(dir?.let { File(it, entry.thumb).path })
                            Column {
                                Text(title, style = MaterialTheme.typography.bodyLarge)
                                Text(
                                    itemDetail(entry),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun Thumbnail(path: String?) {
    val bitmap by produceState<android.graphics.Bitmap?>(initialValue = null, path) {
        value = path?.let { p -> withContext(Dispatchers.IO) { runCatching { BitmapFactory.decodeFile(p) }.getOrNull() } }
    }
    Box(Modifier.size(48.dp).background(Color(0xFF333333), RoundedCornerShape(6.dp))) {
        bitmap?.let { Image(it.asImageBitmap(), contentDescription = null, modifier = Modifier.fillMaxSize()) }
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

/** Callbacks from viewer.js. Only the annotated methods are visible to the page. */
private class ViewerBridge(
    private val onLoadedMain: (Int) -> Unit,
    private val onErrorMain: (String) -> Unit,
    private val onRoamMain: (Boolean) -> Unit = {},
) {
    private val main = Handler(Looper.getMainLooper())

    @JavascriptInterface
    fun onRoam(on: Boolean) {
        main.post { onRoamMain(on) }
    }

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
    objectsPath: () -> String?,
    home: HomeView?,
    onLoaded: (Int) -> Unit,
    onError: (String) -> Unit,
    onRoam: (Boolean) -> Unit = {},
): WebView {
    if (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
        WebView.setWebContentsDebuggingEnabled(true) // chrome://inspect for manual flyTo() tests
    }
    val loader = WebViewAssetLoader.Builder()
        .setDomain(ASSET_HOST)
        .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(context))
        .addPathHandler("/scene/") { path ->
            ScenePathPolicy.resolve(path, scenePath, objectsPath())?.let {
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
        addJavascriptInterface(ViewerBridge(onLoaded, onError, onRoam), "Android")
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
