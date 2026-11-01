package dev.recon3d.nav

import android.widget.Toast
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dev.recon3d.capture.CaptureScreen
import dev.recon3d.net.Api
import dev.recon3d.settings.ServerSettings
import dev.recon3d.settings.SettingsScreen

object Routes {
    const val SETTINGS = "settings"
    const val CAPTURE = "capture/{roomId}"
    fun capture(roomId: String) = "capture/$roomId"
}

/** App navigation graph: settings + AR capture (rooms list: Task 12). */
@Composable
fun AppNav(
    settings: ServerSettings,
    apiFor: (String) -> Api,
    modifier: Modifier = Modifier,
) {
    val nav = rememberNavController()
    val context = LocalContext.current
    NavHost(navController = nav, startDestination = Routes.SETTINGS, modifier = modifier) {
        composable(Routes.SETTINGS) {
            // TEMP (Task 12 replaces this with the rooms list): entry point to the capture screen.
            SettingsScreen(settings = settings, apiFor = apiFor, onOpenCapture = { nav.navigate(Routes.capture("room-test")) })
        }
        composable(Routes.CAPTURE) { entry ->
            val roomId = entry.arguments?.getString("roomId") ?: return@composable
            CaptureScreen(roomId = roomId, onFinished = { zip ->
                Toast.makeText(context, "已保存：${zip.name}", Toast.LENGTH_LONG).show()
                nav.popBackStack()
            })
        }
    }
}
