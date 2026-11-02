package dev.recon3d.capture

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.File
import java.time.Instant
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.temporal.ChronoUnit
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** Writes a capture package (docs/capture-format.md, format v2). Frames are staged in [dir], then zipped by [finish]. */
class CapturePackageWriter(
    private val dir: File,
    private val roomId: String,
    private val captureId: String,
    private val imageW: Int,
    private val imageH: Int,
    private val depthW: Int?,
    private val depthH: Int?,
    private val device: String,
    private val createdAt: String = Instant.now().truncatedTo(ChronoUnit.SECONDS).toString(),
) {
    private var count = 0
    private var finished = false
    private val framesFile = File(dir, "frames.jsonl")

    init {
        File(dir, "images").mkdirs(); File(dir, "depth").mkdirs(); File(dir, "conf").mkdirs()
        framesFile.writeText("")
    }

    fun addFrame(
        tNs: Long, jpeg: ByteArray, depthMm: ShortArray?, poseGl: FloatArray,
        fx: Float, fy: Float, cx: Float, cy: Float,
        depthIntrinsics: FloatArray? = null,
        depthKind: String = "smooth",
        confidence: ByteArray? = null,
    ) {
        require(poseGl.size == 16) { "pose must have 16 floats" }
        require(depthIntrinsics == null || depthIntrinsics.size == 4) { "depthIntrinsics must be [fx, fy, cx, cy]" }
        require(depthKind == "raw" || depthKind == "smooth") { "depthKind must be raw or smooth" }
        require(confidence == null || depthMm != null) { "confidence given without depth" }
        val index = ++count
        val name = "%06d".format(index)
        val imageRel = "images/$name.jpg"
        File(dir, imageRel).writeBytes(jpeg)
        var depthRel: String? = null
        if (depthMm != null) {
            require(depthW != null && depthH != null) { "depth given but depth size not configured" }
            depthRel = "depth/$name.png"
            File(dir, depthRel).writeBytes(Png16.encode(depthW, depthH, depthMm))
        }
        var confRel: String? = null
        if (confidence != null) {
            require(confidence.size == depthW!! * depthH!!) { "confidence size != depthW*depthH" }
            confRel = "conf/$name.png"
            File(dir, confRel).writeBytes(Png8.encode(depthW, depthH, confidence))
        }
        val line = buildJsonObject {
            put("index", index)
            put("t_ns", tNs)
            put("image", imageRel)
            put("depth", depthRel?.let { JsonPrimitive(it) } ?: JsonNull)
            put("pose", JsonArray(poseGl.map { JsonPrimitive(it) }))
            put("fx", fx); put("fy", fy); put("cx", cx); put("cy", cy)
            // Optional (format v1): depth-image intrinsics, written only when this frame has depth.
            if (depthRel != null && depthIntrinsics != null) {
                put("depth_fx", depthIntrinsics[0]); put("depth_fy", depthIntrinsics[1])
                put("depth_cx", depthIntrinsics[2]); put("depth_cy", depthIntrinsics[3])
            }
            // Format v2: depth source, written only when this frame has depth; confidence map path or null.
            if (depthRel != null) put("depth_kind", depthKind)
            put("confidence", confRel?.let { JsonPrimitive(it) } ?: JsonNull)
        }
        framesFile.appendText(line.toString() + "\n")
    }

    /** Writes manifest.json, zips everything (fast deflate; JPEG/PNG are already compressed), deletes staging dir. */
    fun finish(): File {
        check(!finished) { "CapturePackageWriter.finish() already called" }
        finished = true
        val manifest: JsonObject = buildJsonObject {
            put("format_version", 2)
            put("capture_id", captureId)
            put("room_id", roomId)
            put("kind", "model")
            put("device", device)
            put("created_at", createdAt)
            put("image_width", imageW); put("image_height", imageH)
            put("depth_width", depthW?.let { JsonPrimitive(it) } ?: JsonNull)
            put("depth_height", depthH?.let { JsonPrimitive(it) } ?: JsonNull)
            put("frame_count", count)
        }
        File(dir, "manifest.json").writeText(manifest.toString())

        val zip = File(dir.parentFile, "$captureId.zip")
        try {
            ZipOutputStream(zip.outputStream().buffered()).use { zos ->
                zos.setLevel(Deflater.BEST_SPEED)
                val root = dir.toPath()
                dir.walkTopDown().filter { it.isFile }.sortedBy { it.path }.forEach { f ->
                    zos.putNextEntry(ZipEntry(root.relativize(f.toPath()).toString().replace(File.separatorChar, '/')).apply { time = ZIP_ENTRY_TIME_MS })
                    f.inputStream().use { it.copyTo(zos) }
                    zos.closeEntry()
                }
            }
        } catch (e: Throwable) {
            zip.delete(); dir.deleteRecursively()
            throw e
        }
        dir.deleteRecursively()
        return zip
    }

    private companion object {
        /** Fixed entry time 2026-01-01 00:00 (zone-independent DOS fields) so identical content yields identical zip bytes. */
        val ZIP_ENTRY_TIME_MS: Long = LocalDateTime.of(2026, 1, 1, 0, 0).atZone(ZoneId.systemDefault()).toInstant().toEpochMilli()
    }
}
