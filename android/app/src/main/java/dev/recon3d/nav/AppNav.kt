package dev.recon3d.nav

import android.widget.Toast
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dev.recon3d.AppGraph
import dev.recon3d.MSG_NO_SERVER
import dev.recon3d.capture.CaptureScreen
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.RoomListScreen
import dev.recon3d.rooms.RoomScreen
import dev.recon3d.settings.SettingsScreen
import dev.recon3d.viewer.ViewerScreen
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

object Routes {
    const val ROOMS = "rooms"
    const val ROOM = "room/{roomId}"
    const val SETTINGS = "settings"
    const val CAPTURE = "capture/{roomId}"
    const val VIEWER = "viewer/{roomId}/{captureId}"
    fun room(roomId: String) = "room/$roomId"
    fun capture(roomId: String) = "capture/$roomId"
    fun viewer(roomId: String, captureId: String) = "viewer/$roomId/$captureId"
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
                onBack = { nav.popBackStack() },
            )
        }
        composable(Routes.SETTINGS) {
            SettingsScreen(settings = graph.settings, apiFor = graph::apiFor)
        }
        composable(Routes.CAPTURE) { entry ->
            val roomId = entry.arguments?.getString("roomId") ?: return@composable
            CaptureScreen(roomId = roomId, onFinished = { zip ->
                graph.rooms.addCapture(roomId, CaptureRecord(id = zip.nameWithoutExtension, zipPath = zip.path))
                Toast.makeText(context, "已保存拍摄，可以上传了", Toast.LENGTH_LONG).show()
                nav.popBackStack() // back to the room
            })
        }
        composable(Routes.VIEWER) { entry ->
            val roomId = entry.arguments?.getString("roomId") ?: return@composable
            val captureId = entry.arguments?.getString("captureId") ?: return@composable
            ViewerScreen(repo = graph.rooms, roomId = roomId, captureId = captureId)
        }
    }
}
