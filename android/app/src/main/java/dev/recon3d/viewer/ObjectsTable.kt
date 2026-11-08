package dev.recon3d.viewer

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

/** One row of the server's `objects.json` (docs/api.md, GET /jobs/{id}/objects.zip). */
@Serializable
data class ObjectEntry(
    val id: Int,
    val label: String,
    @SerialName("label_zh") val labelZh: String,
    val score: Float,
    val views: Int,
    /** Scene world frame (same as `scene.splat`, +Y up). */
    val center: List<Float>,
    val bbox: Bbox,
    @SerialName("gaussian_count") val gaussianCount: Int,
    /** Relative to the objects directory, e.g. `thumbs/1.jpg`. */
    val thumb: String,
    // v2 fields (docs/api.md); a version-1 table has none of them and lists everything as an item.
    val level: String = "item",
    /** Gravity-aligned length, width, height in metres. */
    val size: List<Float>? = null,
    /** `floor` / `ceiling` / `on:<id>` / `wall:<id>` / `none`. */
    val support: String? = null,
    val confidence: Float? = null,
)

@Serializable
data class Bbox(val min: List<Float>, val max: List<Float>)

/** The object table downloaded for one capture; lives in `<scenesDir>/<room>/<capture>/objects/`. */
@Serializable
data class ObjectsTable(
    val version: Int,
    @SerialName("job_id") val jobId: String,
    val objects: List<ObjectEntry>,
) {
    companion object {
        private val json = Json { ignoreUnknownKeys = true }

        const val JSON_NAME = "objects.json"
        const val BIN_NAME = "objects.bin"

        fun load(dir: File): ObjectsTable = json.decodeFromString(File(dir, JSON_NAME).readText())
    }
}

/** Objects of one Chinese label, as listed on the phone. */
data class ObjectGroup(val labelZh: String, val items: List<ObjectEntry>)

/** Groups by Chinese label: biggest group first (then by name), items by id. */
fun ObjectsTable.grouped(): List<ObjectGroup> =
    objects.groupBy { it.labelZh }
        .map { (label, items) -> ObjectGroup(label, items.sortedBy { it.id }) }
        .sortedWith(compareByDescending<ObjectGroup> { it.items.size }.thenBy { it.labelZh })

/** One level of the list: structure 「结构」, furniture 「家具」, item 「物品」. */
data class LevelSection(val level: String, val title: String, val groups: List<ObjectGroup>)

private val LEVELS = listOf("structure" to "结构", "furniture" to "家具", "item" to "物品")

/** Sections in structure / furniture / item order, each grouped like [grouped]; empty sections are omitted. */
fun ObjectsTable.sections(): List<LevelSection> =
    LEVELS.mapNotNull { (level, title) ->
        val groups = ObjectsTable(version, jobId, objects.filter { (if (it.level in LEVELS.map { l -> l.first }) it.level else "item") == level }).grouped()
        if (groups.isEmpty()) null else LevelSection(level, title, groups)
    }

fun groupTitle(group: ObjectGroup): String = "${group.labelZh} ×${group.items.size}"

fun itemTitle(group: ObjectGroup, index: Int): String = "${group.labelZh} ${index + 1}"

/** Second line of a list item: frames the object was tracked in, plus its height for non-structure objects. */
fun itemDetail(entry: ObjectEntry): String {
    val frames = "出现 ${entry.views} 帧"
    val h = entry.size?.getOrNull(2)
    return if (h != null && entry.level != "structure") frames + " · 高 " + String.format(java.util.Locale.ROOT, "%.2f", h) + " 米" else frames
}

/** The JS call that flies the viewer to [entry] and highlights it (numbers via kotlinx, never locale formatted). */
fun flyToObjectJs(entry: ObjectEntry): String {
    fun v(c: List<Float>) = """{"x":${c[0]},"y":${c[1]},"z":${c[2]}}"""
    return "window.flyToObject(${entry.id}, ${v(entry.center)}, ${v(entry.bbox.min)}, ${v(entry.bbox.max)})"
}
