package dev.recon3d

import android.content.Context
import dev.recon3d.net.Api
import dev.recon3d.net.ApiClient
import dev.recon3d.net.ApiException
import dev.recon3d.net.Health
import dev.recon3d.net.JobStatus
import dev.recon3d.rooms.RoomRepository
import dev.recon3d.settings.ServerSettings
import dev.recon3d.upload.UploadViewModel
import kotlinx.coroutines.flow.first
import java.io.File

const val MSG_NO_SERVER = "请先在设置中填写计算端地址"

/**
 * Manual DI: one instance per process (so uploads keep running across activity recreation).
 * Obtain with [AppGraph.get].
 */
class AppGraph private constructor(context: Context) {
    private val app = context.applicationContext

    val settings = ServerSettings(app)
    val rooms = RoomRepository(app.filesDir)
    val uploader = UploadViewModel(
        api = SettingsBackedApi(settings),
        repo = rooms,
        scenesDir = File(app.filesDir, "rooms"),
    ).also { it.resumeInterrupted() }

    fun apiFor(baseUrl: String): Api = ApiClient(baseUrl)

    companion object {
        @Volatile private var instance: AppGraph? = null

        fun get(context: Context): AppGraph =
            instance ?: synchronized(this) { instance ?: AppGraph(context).also { instance = it } }
    }
}

/** [Api] that builds an [ApiClient] from the address currently saved in [settings] for every call. */
private class SettingsBackedApi(private val settings: ServerSettings) : Api {
    private suspend fun client(): Api =
        ApiClient(settings.baseUrl.first() ?: throw ApiException(MSG_NO_SERVER))

    override suspend fun health(): Health = client().health()
    override suspend fun upload(zip: File, onProgress: (Float) -> Unit): String = client().upload(zip, onProgress)
    override suspend fun job(id: String): JobStatus = client().job(id)
    override suspend fun downloadScene(id: String, dest: File) = client().downloadScene(id, dest)
}
