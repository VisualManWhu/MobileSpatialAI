package dev.recon3d.viewer

import java.io.File

/**
 * Which files the viewer's `/scene/` path handler may serve. Pure JVM so it is unit-tested.
 *
 * The page can only ever reach the ONE scene of the capture being viewed: the request path (the
 * part after `/scene/`) must equal [SCENE_NAME] exactly — no normalisation, so traversal, encoded
 * dots, sub-paths and case tricks all fail — and the bytes always come from the capture's
 * recorded `scenePath`, never from a path derived from the request.
 */
object ScenePathPolicy {
    const val SCENE_NAME = "scene.splat"

    fun allows(requestPath: String): Boolean = requestPath == SCENE_NAME

    /** The file to serve for [requestPath], or null (→ 404) if not allowed or not on disk. */
    fun resolve(requestPath: String, scenePath: String?): File? {
        if (!allows(requestPath) || scenePath == null) return null
        return File(scenePath).takeIf { it.isFile }
    }
}
