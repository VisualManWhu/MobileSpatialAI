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

/** `POST /api/v1/jobs` 201 body. */
@Serializable
internal data class JobCreated(@SerialName("job_id") val jobId: String)

/** Every client-visible failure, already phrased in Chinese for the UI. */
class ApiException(val messageZh: String, cause: Throwable? = null) : Exception(messageZh, cause)
