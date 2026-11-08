package dev.recon3d.viewer

import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ObjectsTableTest {
    @get:Rule val tmp = TemporaryFolder()

    private val json = """
        {"version": 1, "job_id": "abc", "frames_used": 12, "gaussians": 100, "unknown_future_key": 5,
         "objects": [
           {"id": 1, "label": "chair", "label_zh": "椅子", "score": 0.6, "views": 3, "center": [1.0, 0.5, -2.0],
            "bbox": {"min": [0.7, 0.0, -2.3], "max": [1.3, 1.0, -1.7]}, "gaussian_count": 10, "thumb": "thumbs/1.jpg"},
           {"id": 2, "label": "cup", "label_zh": "杯子", "score": 0.7, "views": 2, "center": [0.1, 0.8, -1.0],
            "bbox": {"min": [0.0, 0.75, -1.1], "max": [0.2, 0.9, -0.9]}, "gaussian_count": 4, "thumb": "thumbs/2.jpg"},
           {"id": 3, "label": "chair", "label_zh": "椅子", "score": 0.5, "views": 4, "center": [2.0, 0.5, -2.0],
            "bbox": {"min": [1.7, 0.0, -2.3], "max": [2.3, 1.0, -1.7]}, "gaussian_count": 8, "thumb": "thumbs/3.jpg"}
         ],
         "relations": [{"a": 2, "rel": "on", "b": 1}]}
    """.trimIndent()

    @Test fun `parses the server json ignoring unknown keys`() {
        val dir = tmp.newFolder("objects")
        dir.resolve("objects.json").writeText(json)
        val t = ObjectsTable.load(dir)
        assertEquals(3, t.objects.size)
        assertEquals("椅子", t.objects[0].labelZh)
        assertEquals(listOf(0.7f, 0.0f, -2.3f), t.objects[0].bbox.min)
        assertEquals("thumbs/2.jpg", t.objects[1].thumb)
    }

    @Test fun `groups by chinese label, biggest group first, items by id`() {
        val dir = tmp.newFolder("objects")
        dir.resolve("objects.json").writeText(json)
        val groups = ObjectsTable.load(dir).grouped()
        assertEquals(listOf("椅子", "杯子"), groups.map { it.labelZh })
        assertEquals(listOf(1, 3), groups[0].items.map { it.id })
        assertEquals("椅子 ×2", groupTitle(groups[0]))
        assertEquals("椅子 2", itemTitle(groups[0], 1))
        assertEquals("杯子 1", itemTitle(groups[1], 0))
        // How many sampled frames the object was detected in: a low count flags a likely false positive.
        assertEquals("出现 4 帧", itemDetail(groups[0].items[1]))
        assertEquals("出现 2 帧", itemDetail(groups[1].items[0]))
    }
}

class FlyToObjectJsTest {
    @Test fun `builds the viewer call with locale-safe numbers`() {
        val e = ObjectEntry(7, "cup", "杯子", 0.5f, 2, listOf(1.5f, -0.25f, 2f), Bbox(listOf(1f, -1f, 1f), listOf(2f, 0.5f, 3f)), 4, "thumbs/7.jpg")
        assertEquals(
            """window.flyToObject(7, {"x":1.5,"y":-0.25,"z":2.0}, {"x":1.0,"y":-1.0,"z":1.0}, {"x":2.0,"y":0.5,"z":3.0})""",
            flyToObjectJs(e),
        )
    }
}
