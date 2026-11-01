package dev.recon3d.nav

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dev.recon3d.net.Api
import dev.recon3d.settings.ServerSettings
import dev.recon3d.settings.SettingsScreen

object Routes {
    const val SETTINGS = "settings"
}

/** App navigation graph. Only the settings screen exists so far (rooms list: Task 12). */
@Composable
fun AppNav(
    settings: ServerSettings,
    apiFor: (String) -> Api,
    modifier: Modifier = Modifier,
) {
    val nav = rememberNavController()
    NavHost(navController = nav, startDestination = Routes.SETTINGS, modifier = modifier) {
        composable(Routes.SETTINGS) {
            SettingsScreen(settings = settings, apiFor = apiFor)
        }
    }
}
