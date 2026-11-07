package dev.recon3d.viewer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ScenePathPolicyTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun `allows only the exact scene file name`() {
        assertTrue(ScenePathPolicy.allows("scene.splat"))
    }

    @Test fun `rejects traversal, other files and lookalikes`() {
        listOf(
            "", "/", "/scene.splat", "scene.splat/", "./scene.splat", "../scene.splat",
            "../rooms.json", "../../shared_prefs/x.xml", "..%2Fscene.splat", "%2e%2e/rooms.json",
            "scene.splat/../rooms.json", "sub/scene.splat", "SCENE.SPLAT", "scene.splat ",
            "scene.splat?x=1", "scene.splat#a", "scene.splat\u0000", "scene.ply", "rooms.json",
        ).forEach { assertFalse("must reject '$it'", ScenePathPolicy.allows(it)) }
    }

    @Test fun `resolve returns the capture's file for the allowed path only`() {
        val scene = tmp.newFile("scene.splat").apply { writeBytes(ByteArray(32)) }
        assertEquals(scene, ScenePathPolicy.resolve("scene.splat", scene.path))
        assertNull(ScenePathPolicy.resolve("../scene.splat", scene.path))
        assertNull(ScenePathPolicy.resolve("rooms.json", scene.path))
    }

    @Test fun `resolve returns null when the capture has no downloaded scene`() {
        assertNull(ScenePathPolicy.resolve("scene.splat", null))
        assertNull(ScenePathPolicy.resolve("scene.splat", tmp.root.resolve("missing.splat").path))
        assertNull("a directory is not a scene", ScenePathPolicy.resolve("scene.splat", tmp.root.path))
    }

    @Test fun `resolve serves the recorded file even if it is not named scene_splat on disk`() {
        val stored = tmp.newFile("cap-1.splat").apply { writeBytes(ByteArray(64)) }
        assertEquals(stored, ScenePathPolicy.resolve("scene.splat", stored.path))
    }

    @Test fun `objects_bin is served from the objects directory only when present`() {
        val scene = tmp.newFile("scene.splat").apply { writeBytes(ByteArray(32)) }
        val objects = tmp.newFolder("objects").apply { resolve("objects.bin").writeBytes(ByteArray(4)) }
        assertTrue(ScenePathPolicy.allows("objects.bin"))
        assertEquals(objects.resolve("objects.bin"), ScenePathPolicy.resolve("objects.bin", scene.path, objects.path))
        assertNull(ScenePathPolicy.resolve("objects.bin", scene.path, null))
        assertNull(ScenePathPolicy.resolve("objects.bin", scene.path, tmp.root.resolve("missing").path))
        assertNull(ScenePathPolicy.resolve("../objects.bin", scene.path, objects.path))
        assertNull(ScenePathPolicy.resolve("objects.json", scene.path, objects.path))
        assertEquals(scene, ScenePathPolicy.resolve("scene.splat", scene.path, objects.path))
    }
}
