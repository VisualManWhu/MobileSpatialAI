package dev.recon3d.net

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import okhttp3.Call
import okhttp3.Callback
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okio.BufferedSink
import okio.buffer
import okio.source
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Progress of a resumable upload (see [Api.upload]). Callbacks run on a background thread. */
interface UploadListener {
    /** The server assigned an upload id; persist it at once so a killed process can continue. */
    fun onCreated(uploadId: String)

    /** Fraction 0..1 of bytes the server has confirmed (not bytes merely written to the socket). */
    fun onProgress(fraction: Float)

    /** The [attempt]-th consecutive failure; the client now waits and reconnects. The next [onProgress] means recovery. */
    fun onRetry(attempt: Int)
}

/** Compute-server API (docs/api.md). An interface so screens can be tested with a fake. */
interface Api {
    suspend fun health(): Health

    /**
     * Uploads a capture zip with the resumable protocol and returns the job id. [uploadId] is the id
     * persisted from an interrupted attempt (null = start a new upload); the transfer survives dropped
     * connections by asking the server where to continue.
     */
    suspend fun upload(zip: File, uploadId: String?, listener: UploadListener): String

    suspend fun job(id: String): JobStatus

    /**
     * Downloads `scene.splat` to [dest], resuming a leftover `dest.part` with `Range`; [dest] only
     * appears once the download is complete. [onRetry] gets the consecutive-failure count before each wait.
     */
    suspend fun downloadScene(id: String, dest: File, onRetry: (Int) -> Unit = {})
}

/**
 * OkHttp implementation of [Api]. [baseUrl] is what the user configured, e.g.
 * `http://192.168.1.20:8765` (see `normalizeBaseUrl`); `/api/v1` is appended here.
 * Every failure surfaces as [ApiException] with a Chinese message.
 */
class ApiClient(
    private val baseUrl: String,
    private val http: OkHttpClient = defaultHttpClient,
    /** Backoff wait in ms between transfer retries; injectable so tests do not sleep. */
    private val sleep: suspend (Long) -> Unit = { delay(it) },
) : Api {

    override suspend fun health(): Health =
        call(get("health")) { decode<Health>(it.body!!.string()) }

    override suspend fun upload(zip: File, uploadId: String?, listener: UploadListener): String {
        val size = zip.length()
        val sha = withContext(Dispatchers.IO) { sha256Hex(zip) }
        var id: String? = uploadId
        var received = -1L  // unknown until the server says
        var alreadyJob = false  // a previous run finished; only the id is needed
        var attempt = 0

        // Runs [block] until it succeeds; every failure asks the server where to continue before the
        // next try, so a chunk cut mid-stream resumes from the bytes that actually landed.
        suspend fun <T> retrying(block: suspend () -> T): T {
            while (true) {
                try {
                    return block().also { attempt = 0 }
                } catch (e: ApiException) {
                    if (!e.isRetryable()) throw e
                    attempt++
                    if (attempt > MAX_TRANSFER_RETRIES) throw ApiException(MSG_UNREACHABLE, e)
                    listener.onRetry(attempt)
                    sleep(backoffMs(attempt))
                    received = -1L
                }
            }
        }

        // Where to continue: a known upload's received count, or a fresh registration.
        suspend fun locate() {
            val known = id
            if (known != null) {
                val info = try {
                    call(get("uploads", known)) { decode<UploadInfo>(it.body!!.string()) }
                } catch (e: ApiException) {
                    if (e.code != 404) throw e
                    null
                }
                if (info != null) {
                    received = info.received
                    return
                }
                // 404: finished before the process died (then the job exists), or expired on the server.
                val job = try {
                    call(get("jobs", known)) { decode<JobStatus>(it.body!!.string()) }
                } catch (e: ApiException) {
                    if (e.code != 404) throw e
                    null
                }
                if (job != null) {
                    received = size
                    alreadyJob = true
                    return
                }
                id = null
            }
            val body = json.encodeToString(UploadCreateBody.serializer(), UploadCreateBody(size, sha))
            val created = call(Request.Builder().url(url("uploads")).post(body.toRequestBody(JSON)).build()) {
                decode<UploadCreated>(it.body!!.string())
            }
            id = created.uploadId
            received = created.received
            listener.onCreated(created.uploadId)
        }

        retrying { locate() }
        while (received < size) {
            val outcome = retrying {
                if (received < 0) locate()
                val current = id!!
                val length = minOf(CHUNK_BYTES, size - received)
                val request = Request.Builder()
                    .url(url("uploads", current))
                    .header("Upload-Offset", received.toString())
                    .put(FileRangeBody(zip, received, length))
                    .build()
                try {
                    PutOutcome.Done(call(request) { decode<Received>(it.body!!.string()).received })
                } catch (e: ApiException) {
                    val corrected = e.received
                    if (e.code == 409 && corrected != null && e.messageZh != MSG_BUSY) PutOutcome.Corrected(corrected)
                    else throw e
                }
            }
            received = when (outcome) {
                is PutOutcome.Done -> outcome.received
                is PutOutcome.Corrected -> outcome.received
            }
            listener.onProgress((received.toFloat() / size).coerceIn(0f, 1f))
        }
        if (alreadyJob) {
            listener.onProgress(1f)
            return id!!
        }
        val jobId = retrying {
            if (received < 0) locate()
            val current = id!!
            try {
                call(Request.Builder().url(url("uploads", current, "finish")).post(EMPTY).build()) {
                    decode<JobCreated>(it.body!!.string()).jobId
                }
            } catch (e: ApiException) {
                if (e.code == 404) current else throw e  // GET /jobs/{id} already proved the job exists
            }
        }
        listener.onProgress(1f)
        return jobId
    }

    override suspend fun job(id: String): JobStatus =
        call(get("jobs", id)) { decode<JobStatus>(it.body!!.string()) }

    override suspend fun downloadScene(id: String, dest: File, onRetry: (Int) -> Unit) {
        dest.absoluteFile.parentFile?.mkdirs()
        val part = File(dest.path + ".part")
        var attempt = 0
        var allowRangeRetry = true
        while (true) {
            val have = if (part.isFile) part.length() else 0L
            val builder = Request.Builder().url(url("jobs", id, "scene.splat")).get()
            if (have > 0) builder.header("Range", "bytes=$have-")
            try {
                call(builder.build()) { response ->
                    val append = response.code == 206
                    response.body!!.source().use { src ->
                        java.io.FileOutputStream(part, append).use { out -> src.inputStream().copyTo(out) }
                    }
                }
                if (dest.exists()) dest.delete()
                if (!part.renameTo(dest)) throw IOException("rename ${part.path} -> ${dest.path} failed")
                return
            } catch (e: ApiException) {
                if (e.code == 416 && allowRangeRetry) {
                    // The server does not agree with our part (different file): start over once.
                    allowRangeRetry = false
                    part.delete()
                    continue
                }
                if (!e.isRetryable()) {
                    if (e.code != null) part.delete()
                    throw e
                }
                attempt++
                if (attempt > MAX_TRANSFER_RETRIES) throw ApiException(MSG_UNREACHABLE, e)
                onRetry(attempt)
                sleep(backoffMs(attempt))
            } catch (e: IOException) {
                attempt++
                if (attempt > MAX_TRANSFER_RETRIES) throw ApiException(MSG_UNREACHABLE, e)
                onRetry(attempt)
                sleep(backoffMs(attempt))
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
        var received: Long? = null
        val detail = try {
            val obj = json.parseToJsonElement(response.body?.string().orEmpty()) as? JsonObject
            received = obj?.get("received")?.jsonPrimitive?.longOrNull
            (obj?.get("detail") as? JsonPrimitive)?.takeIf { it.isString }?.content
        } catch (_: Exception) {
            null
        }
        return ApiException(detail ?: "计算端返回错误：HTTP ${response.code}", code = response.code)
            .also { it.received = received }
    }

    private inline fun <reified T> decode(text: String): T = json.decodeFromString(text)

    private sealed interface PutOutcome {
        class Done(val received: Long) : PutOutcome
        class Corrected(val received: Long) : PutOutcome
    }

    @kotlinx.serialization.Serializable
    private data class UploadCreateBody(val size: Long, val sha256: String)

    companion object {
        private const val TAG = "ApiClient"
        const val MSG_UNREACHABLE = "连不上计算端，请检查 Wi-Fi 和地址"
        const val MSG_UNPARSEABLE = "计算端返回的数据无法解析"
        const val MSG_BAD_ADDRESS = "计算端地址格式不正确"
        /** Server detail for a PUT that overlaps another one (docs/api.md); retried like a drop. */
        const val MSG_BUSY = "该上传正在接收中"

        /** Bytes per PUT; the server keeps whatever arrives even when a chunk is cut short. */
        const val CHUNK_BYTES = 4L * 1024 * 1024
        /** Reconnect attempts after consecutive failures before an upload/download gives up (~15 min of backoff). */
        const val MAX_TRANSFER_RETRIES = 30
        private const val BACKOFF_BASE_MS = 2_000L
        private const val BACKOFF_MAX_MS = 30_000L

        /** n-th consecutive failure: 2 s × 2^(n−1), capped at 30 s. */
        fun backoffMs(attempt: Int): Long =
            (BACKOFF_BASE_MS shl (attempt - 1).coerceIn(0, 20)).coerceAtMost(BACKOFF_MAX_MS)

        private val JSON: MediaType = "application/json".toMediaType()
        private val OCTET: MediaType = "application/octet-stream".toMediaType()
        private val EMPTY: RequestBody = ByteArray(0).toRequestBody(null)
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

/** Transport failures, server 5xx and the busy 409 are worth waiting and trying again. */
private fun ApiException.isRetryable(): Boolean =
    code == null || code >= 500 || (code == 409 && messageZh == ApiClient.MSG_BUSY)

/** `received` from a 409 body, carried next to the exception without widening its public constructor. */
private val receivedByException = java.util.WeakHashMap<ApiException, Long>()
private var ApiException.received: Long?
    get() = synchronized(receivedByException) { receivedByException[this] }
    set(value) { synchronized(receivedByException) { if (value != null) receivedByException[this] = value } }

/** Lowercase hex SHA-256 of [file], streamed in 1 MB blocks. */
internal fun sha256Hex(file: File): String {
    val md = MessageDigest.getInstance("SHA-256")
    FileInputStream(file).use { input ->
        val buf = ByteArray(1024 * 1024)
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            md.update(buf, 0, n)
        }
    }
    return md.digest().joinToString("") { "%02x".format(it) }
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

/** [length] bytes of [file] from [offset], streamed from disk; one-shot so OkHttp never resends it on its own. */
private class FileRangeBody(private val file: File, private val offset: Long, private val length: Long) : RequestBody() {
    override fun contentType(): MediaType = "application/octet-stream".toMediaType()
    override fun contentLength(): Long = length
    override fun isOneShot(): Boolean = true

    override fun writeTo(sink: BufferedSink) {
        FileInputStream(file).use { input ->
            var skipped = 0L
            while (skipped < offset) {
                val n = input.skip(offset - skipped)
                if (n <= 0) throw IOException("cannot seek to $offset in ${file.path}")
                skipped += n
            }
            input.source().buffer().use { src ->
                var remaining = length
                while (remaining > 0) {
                    val n = src.read(sink.buffer, minOf(remaining, 64L * 1024))
                    if (n < 0) throw IOException("${file.path} shorter than expected")
                    remaining -= n
                    sink.emitCompleteSegments()
                }
            }
        }
    }
}
