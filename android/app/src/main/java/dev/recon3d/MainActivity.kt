package dev.recon3d

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import dev.recon3d.nav.AppNav
import dev.recon3d.net.ApiClient
import dev.recon3d.settings.ServerSettings

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val settings = ServerSettings(applicationContext)
        setContent {
            // minSdk 33 ≥ 31, so Material You dynamic color is always available.
            val context = LocalContext.current
            val colors = if (isSystemInDarkTheme()) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
            MaterialTheme(colorScheme = colors) {
                Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
                    AppNav(
                        settings = settings,
                        apiFor = { baseUrl -> ApiClient(baseUrl) },
                        modifier = Modifier.padding(padding),
                    )
                }
            }
        }
    }
}
