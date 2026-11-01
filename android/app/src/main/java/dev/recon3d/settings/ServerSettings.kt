package dev.recon3d.settings

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

// One DataStore per process (the delegate guarantees a singleton per name).
private val Context.serverSettingsStore: DataStore<Preferences> by preferencesDataStore("server_settings")

/** Persists the compute-server address the user typed on the settings screen. */
class ServerSettings(context: Context) {
    private val store = context.applicationContext.serverSettingsStore

    /** Normalized base URL (e.g. `http://192.168.1.20:8765`), or null if never set. */
    val baseUrl: Flow<String?> = store.data.map { it[KEY_BASE_URL] }

    /** Saves [url] after [normalizeBaseUrl]; throws [IllegalArgumentException] if blank. */
    suspend fun setBaseUrl(url: String) {
        val normalized = normalizeBaseUrl(url)
        store.edit { it[KEY_BASE_URL] = normalized }
    }

    private companion object {
        val KEY_BASE_URL = stringPreferencesKey("base_url")
    }
}

/**
 * Turns what a user types into a base URL: trims, adds `http://` when there is no scheme,
 * drops trailing slashes. Any host form is accepted — LAN IPs (`192.168.1.20:8765`),
 * Tailscale IPs (`100.101.102.103:8765`), host names.
 */
fun normalizeBaseUrl(input: String): String {
    var s = input.trim()
    require(s.isNotEmpty()) { "server address is blank" }
    if (!s.contains("://")) s = "http://$s"
    return s.trimEnd('/')
}
