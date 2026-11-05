package dev.recon3d.net

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** `GET /api/v1/health` — see docs/api.md. */
@Serializable
data class Health(
    val version: String,
    val gpu: String,
    /** `"laptop"` or `"server"`. */
    val profile: String,
)

/** `GET /api/v1/jobs/{id}` — see docs/api.md. */
@Serializable
data class JobStatus(
    @SerialName("job_id") val jobId: String,
    val state: JobState,
    /** 0..1 (training 0→0.9, export 0.9→1.0). */
    val progress: Float,
    /** Chinese status text; on failure, the reason. */
    val message: String,
)

@Serializable
enum class JobState {
    @SerialName("queued") QUEUED,
    @SerialName("training") TRAINING,
    @SerialName("exporting") EXPORTING,
    @SerialName("done") DONE,
    @SerialName("failed") FAILED,
}

/** `POST /api/v1/jobs` and `POST /api/v1/uploads/{id}/finish` 201 body. */
@Serializable
internal data class JobCreated(@SerialName("job_id") val jobId: String)

/** `POST /api/v1/uploads` 201 body. */
@Serializable
internal data class UploadCreated(@SerialName("upload_id") val uploadId: String, val received: Long)

/** `GET /api/v1/uploads/{id}` body. */
@Serializable
internal data class UploadInfo(@SerialName("upload_id") val uploadId: String, val size: Long, val received: Long)

/** `PUT /api/v1/uploads/{id}` 200 body; also the extra key on its 409 bodies. */
@Serializable
internal data class Received(val received: Long)

/**
 * Every client-visible failure, already phrased in Chinese for the UI. [code] is the HTTP status for a
 * server error response, null for transport failures (unreachable, dropped, timed out).
 */
class ApiException(val messageZh: String, cause: Throwable? = null, val code: Int? = null) : Exception(messageZh, cause)
