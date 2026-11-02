package dev.recon3d.net

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import okhttp3.Call
import okhttp3.Callback
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.Response
import okio.BufferedSink
import okio.ForwardingSink
import okio.Sink
import okio.buffer
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Compute-server API (docs/api.md). An interface so screens can be tested with a fake. */
interface Api {
    suspend fun health(): Health

    /** Uploads a capture zip; [onProgress] gets 0..1 as bytes are sent. Returns the job id. */
    suspend fun upload(zip: File, onProgress: (Float) -> Unit): String

    suspend fun job(id: String): JobStatus

    /** Downloads `scene.splat` to [dest]; [dest] only appears once the download is complete. */
    suspend fun downloadScene(id: String, dest: File)
}

/**
 * OkHttp implementation of [Api]. [baseUrl] is what the user configured, e.g.
 * `http://192.168.1.20:8765` (see `normalizeBaseUrl`); `/api/v1` is appended here.
 * Every failure surfaces as [ApiException] with a Chinese message.
 */
class ApiClient(
    private val baseUrl: String,
    private val http: OkHttpClient = defaultHttpClient,
) : Api {

    override suspend fun health(): Health =
        call(get("health")) { decode<Health>(it.body!!.string()) }

    override suspend fun upload(zip: File, onProgress: (Float) -> Unit): String {
        val file = ProgressRequestBody(zip.asRequestBody(ZIP), onProgress)
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("capture", zip.name, file)
            .build()
        val request = Request.Builder().url(url("jobs")).post(body).build()
        return call(request) { decode<JobCreated>(it.body!!.string()).jobId }
    }

    override suspend fun job(id: String): JobStatus =
        call(get("jobs", id)) { decode<JobStatus>(it.body!!.string()) }

    override suspend fun downloadScene(id: String, dest: File) {
        call(get("jobs", id, "scene.splat")) { response ->
            dest.absoluteFile.parentFile?.mkdirs()
            val part = File(dest.path + ".part")
            try {
                response.body!!.source().use { src ->
                    part.outputStream().use { out -> src.inputStream().copyTo(out) }
                }
                if (dest.exists()) dest.delete()
                if (!part.renameTo(dest)) throw IOException("rename ${part.path} -> ${dest.path} failed")
            } finally {
                part.delete()
            }
        }
    }

    private fun get(vararg segments: String) = Request.Builder().url(url(*segments)).get().build()

    private fun url(vararg segments: String): HttpUrl {
        val base = baseUrl.toHttpUrlOrNull() ?: throw ApiException(MSG_BAD_ADDRESS)
        return base.newBuilder()
            .addPathSegments("api/v1")
            .apply { segments.forEach { addPathSegment(it) } }
            .build()
    }

    /** Executes [request] off the main thread and maps every failure to [ApiException]. */
    private suspend fun <T> call(request: Request, onSuccess: (Response) -> T): T =
        withContext(Dispatchers.IO) {
            try {
                http.newCall(request).await().use { response ->
                    if (!response.isSuccessful) throw httpError(response)
                    onSuccess(response)
                }
            } catch (e: ApiException) {
                throw e
            } catch (e: SerializationException) {
                throw ApiException(MSG_UNPARSEABLE, e)
            } catch (e: IllegalArgumentException) {
                // kotlinx.serialization also reports some malformed input this way.
                throw ApiException(MSG_UNPARSEABLE, e)
            } catch (e: IOException) {
                // Refused, unknown host, timeout, no route, connection dropped mid-transfer.
                Log.w(TAG, "request failed: ${request.method} ${request.url}", e)
                throw ApiException(MSG_UNREACHABLE, e)
            }
        }

    private fun httpError(response: Response): ApiException {
        val detail = try {
            val obj = json.parseToJsonElement(response.body?.string().orEmpty()) as? JsonObject
            (obj?.get("detail") as? JsonPrimitive)?.takeIf { it.isString }?.content
        } catch (_: Exception) {
            null
        }
        return ApiException(detail ?: "计算端返回错误：HTTP ${response.code}")
    }

    private inline fun <reified T> decode(text: String): T = json.decodeFromString(text)

    companion object {
        private const val TAG = "ApiClient"
        const val MSG_UNREACHABLE = "连不上计算端，请检查 Wi-Fi 和地址"
        const val MSG_UNPARSEABLE = "计算端返回的数据无法解析"
        const val MSG_BAD_ADDRESS = "计算端地址格式不正确"

        private val ZIP: MediaType = "application/zip".toMediaType()
        private val json = Json { ignoreUnknownKeys = true }

        /** Shared client: short connect timeout so a wrong address fails fast; uploads stream. */
        val defaultHttpClient: OkHttpClient by lazy {
            OkHttpClient.Builder()
                .connectTimeout(5, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .build()
        }
    }
}

/** Cancels the HTTP call when the coroutine is cancelled. */
private suspend fun Call.await(): Response = suspendCancellableCoroutine { cont ->
    cont.invokeOnCancellation { cancel() }
    enqueue(object : Callback {
        override fun onResponse(call: Call, response: Response) {
            cont.resume(response) { _, value, _ -> value.close() }
        }

        override fun onFailure(call: Call, e: IOException) {
            cont.resumeWithException(e)
        }
    })
}

/** Wraps [delegate] and reports the fraction of its bytes written so far. */
private class ProgressRequestBody(
    private val delegate: RequestBody,
    private val onProgress: (Float) -> Unit,
) : RequestBody() {
    override fun contentType() = delegate.contentType()
    override fun contentLength() = delegate.contentLength()

    override fun writeTo(sink: BufferedSink) {
        val total = contentLength()
        var written = 0L
        var lastReported = -1f
        val counting = object : ForwardingSink(sink) {
            override fun write(source: okio.Buffer, byteCount: Long) {
                super.write(source, byteCount)
                written += byteCount
                if (total > 0) {
                    val p = (written.toFloat() / total).coerceIn(0f, 1f)
                    // Throttle to ~1% steps so a 4 GB upload doesn't flood the UI.
                    if (p - lastReported >= 0.01f || p == 1f && lastReported < 1f) {
                        lastReported = p
                        onProgress(p)
                    }
                }
            }
        }
        val buffered = (counting as Sink).buffer()
        delegate.writeTo(buffered)
        buffered.flush()
        if (total <= 0) onProgress(1f)
    }
}
