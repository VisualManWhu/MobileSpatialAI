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

    @Test fun `version 1 table lists everything under 物品`() {
        val dir = tmp.newFolder("objects")
        dir.resolve("objects.json").writeText(json)
        val sections = ObjectsTable.load(dir).sections()
        assertEquals(listOf("物品"), sections.map { it.title })
        assertEquals("item", sections[0].level)
        assertEquals(listOf("椅子", "杯子"), sections[0].groups.map { it.labelZh })
    }

    private val jsonV2 = """
        {"version": 2, "job_id": "abc", "frames_used": 1100, "gaussians": 100,
         "objects": [
           {"id": 1, "label": "cup", "label_zh": "杯子", "score": 0.7, "views": 20, "center": [0.1, 0.8, -1.0],
            "bbox": {"min": [0.0, 0.75, -1.1], "max": [0.2, 0.9, -0.9]}, "gaussian_count": 4, "thumb": "thumbs/1.jpg",
            "level": "item", "size": [0.1, 0.08, 0.12], "support": "on:2", "confidence": 0.9},
           {"id": 2, "label": "chair", "label_zh": "椅子", "score": 0.6, "views": 300, "center": [1.0, 0.5, -2.0],
            "bbox": {"min": [0.7, 0.0, -2.3], "max": [1.3, 1.0, -1.7]}, "gaussian_count": 10, "thumb": "thumbs/2.jpg",
            "level": "furniture", "size": [0.6, 0.55, 0.904], "support": "floor", "confidence": 0.8},
           {"id": 3, "label": "door", "label_zh": "门", "score": 1.0, "views": 80, "center": [-2.0, 1.0, 0.0],
            "bbox": {"min": [-2.1, 0.0, -0.5], "max": [-1.9, 2.0, 0.5]}, "gaussian_count": 8, "thumb": "thumbs/3.jpg",
            "level": "structure", "size": [1.0, 0.1, 2.0], "support": "wall:1", "confidence": 1.0},
           {"id": 4, "label": "floor", "label_zh": "地面", "score": 1.0, "views": 1100, "center": [0.0, 0.0, 0.0],
            "bbox": {"min": [-2.0, -0.05, -2.5], "max": [2.0, 0.05, 2.5]}, "gaussian_count": 800, "thumb": "thumbs/4.jpg",
            "level": "structure", "size": [4.0, 5.0, 0.1], "support": "none", "confidence": 1.0}
         ],
         "relations": [], "structure": {"floor_y": 0.0, "ceiling_y": null, "walls": []}}
    """.trimIndent()

    @Test fun `version 2 table has structure, furniture and item sections in that order`() {
        val dir = tmp.newFolder("objects")
        dir.resolve("objects.json").writeText(jsonV2)
        val t = ObjectsTable.load(dir)
        val sections = t.sections()
        assertEquals(listOf("结构", "家具", "物品"), sections.map { it.title })
        assertEquals(listOf("structure", "furniture", "item"), sections.map { it.level })
        assertEquals(listOf("地面", "门"), sections[0].groups.map { it.labelZh })  // equal counts -> by name
        val chair = t.objects.first { it.label == "chair" }
        assertEquals("出现 300 帧 · 高 0.90 米", itemDetail(chair))
        assertEquals("出现 80 帧", itemDetail(t.objects.first { it.label == "door" }))  // structure: no size line
        assertEquals("on:2", chair.let { t.objects.first { o -> o.label == "cup" }.support })
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
