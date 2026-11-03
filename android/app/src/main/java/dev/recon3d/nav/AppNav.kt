package dev.recon3d.nav

import android.widget.Toast
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.Lifecycle
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dev.recon3d.AppGraph
import dev.recon3d.MSG_NO_SERVER
import dev.recon3d.capture.CaptureScreen
import dev.recon3d.capture.PreviewScreen
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.RoomListScreen
import dev.recon3d.rooms.RoomScreen
import dev.recon3d.settings.SettingsScreen
import dev.recon3d.viewer.ViewerScreen
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.File

object Routes {
    const val ROOMS = "rooms"
    const val ROOM = "room/{roomId}"
    const val SETTINGS = "settings"
    const val CAPTURE = "capture/{roomId}"
    const val VIEWER = "viewer/{roomId}/{captureId}"
    const val PREVIEW = "preview/{roomId}/{captureId}"
    /** Read-only point cloud of a capture that is uploading / training (opened from the room). */
    const val POINTS = "points/{roomId}/{captureId}"
    fun room(roomId: String) = "room/$roomId"
    fun capture(roomId: String) = "capture/$roomId"
    fun viewer(roomId: String, captureId: String) = "viewer/$roomId/$captureId"
    fun preview(roomId: String, captureId: String) = "preview/$roomId/$captureId"
    fun points(roomId: String, captureId: String) = "points/$roomId/$captureId"
}

/** App navigation graph: rooms (start) → room → capture / viewer; gear → settings. */
@Composable
fun AppNav(graph: AppGraph, modifier: Modifier = Modifier) {
    val nav = rememberNavController()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    /** Runs [action] only if a server address is saved; otherwise asks for one in settings. */
    fun withServer(action: () -> Unit) {
        scope.launch {
            if (graph.settings.baseUrl.first() == null) {
                Toast.makeText(context, MSG_NO_SERVER, Toast.LENGTH_LONG).show()
                nav.navigate(Routes.SETTINGS)
            } else {
                action()
            }
        }
    }

    NavHost(navController = nav, startDestination = Routes.ROOMS, modifier = modifier) {
        composable(Routes.ROOMS) {
            RoomListScreen(
                repo = graph.rooms,
                onOpenRoom = { nav.navigate(Routes.room(it)) },
                onOpenSettings = { nav.navigate(Routes.SETTINGS) },
            )
        }
        composable(Routes.ROOM) { entry ->
            val roomId = entry.arguments?.getString("roomId") ?: return@composable
            RoomScreen(
                repo = graph.rooms,
                roomId = roomId,
                onCapture = { nav.navigate(Routes.capture(roomId)) },
                onUpload = { id -> withServer { graph.uploader.start(roomId, id) } },
                onRetry = { id -> withServer { graph.uploader.retry(roomId, id) } },
                onView = { id -> nav.navigate(Routes.viewer(roomId, id)) },
                onViewPoints = { id -> nav.navigate(Routes.points(roomId, id)) },
                onBack = { nav.popBackStack() },
            )
        }
        composable(Routes.SETTINGS) {
            SettingsScreen(settings = graph.settings, apiFor = graph::apiFor)
        }
        composable(Routes.CAPTURE) { entry ->
            val roomId = entry.arguments?.getString("roomId") ?: return@composable
            CaptureScreen(roomId = roomId, onFinished = { out ->
                val id = out.zip.nameWithoutExtension
                graph.rooms.addCapture(
                    roomId,
                    CaptureRecord(
                        id = id, zipPath = out.zip.path, pointsPath = out.points?.path,
                        frameCount = out.frames, pointCount = out.pointCount,
                    ),
                )
                // Capture screen leaves the back stack: back from the preview returns to the room.
                nav.navigate(Routes.preview(roomId, id)) { popUpTo(Routes.ROOM) }
            })
        }
        composable(Routes.PREVIEW) { entry ->
            val roomId = entry.arguments?.getString("roomId") ?: return@composable
            val captureId = entry.arguments?.getString("captureId") ?: return@composable
            val record = runCatching { graph.rooms.capture(roomId, captureId) }.getOrNull() ?: return@composable
            // Ignore the second tap of a double tap: once this entry is leaving, its buttons must not navigate again.
            fun resumed() = entry.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)
            PreviewScreen(
                record = record,
                onUpload = {
                    if (!resumed()) return@PreviewScreen
                    withServer { graph.uploader.start(roomId, captureId) }
                    nav.popBackStack() // room shows the upload progress
                },
                onRetake = {
                    if (!resumed()) return@PreviewScreen
                    graph.rooms.removeCapture(roomId, captureId)
                    File(record.zipPath).delete()
                    record.pointsPath?.let { File(it).delete() }
                    nav.navigate(Routes.capture(roomId)) { popUpTo(Routes.ROOM) }
                },
            )
        }
        composable(Routes.POINTS) { entry ->
            val roomId = entry.arguments?.getString("roomId") ?: return@composable
            val captureId = entry.arguments?.getString("captureId") ?: return@composable
            // Observe the repository so the status line follows the upload / training progress.
            val rooms by graph.rooms.rooms().collectAsState()
            val record = rooms.firstOrNull { it.id == roomId }?.captures?.firstOrNull { it.id == captureId }
                ?: return@composable
            PreviewScreen(record = record, onUpload = null, onRetake = null)
        }
        composable(Routes.VIEWER) { entry ->
            val roomId = entry.arguments?.getString("roomId") ?: return@composable
            val captureId = entry.arguments?.getString("captureId") ?: return@composable
            ViewerScreen(repo = graph.rooms, roomId = roomId, captureId = captureId)
        }
    }
}
