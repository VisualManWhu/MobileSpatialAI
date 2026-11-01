/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified for Recon3D (2026): adapted from the ARCore SDK sample
 * samples/hello_ar_kotlin/.../common/samplerender/arcore/BackgroundRenderer.java.
 * Ported to Kotlin, camera-feed path only (no depth visualization / occlusion / virtual scene),
 * and the SampleRender Mesh/Shader/Texture/VertexBuffer abstractions replaced by direct GLES 3.0 calls.
 */
package dev.recon3d.capture.ar

import android.content.res.AssetManager
import android.opengl.GLES11Ext
import android.opengl.GLES30
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** Draws the ARCore camera image as a full-screen background. All methods must run on the GL thread. */
class BackgroundRenderer {
    private val ndcQuad: FloatBuffer = floatBuffer(floatArrayOf(-1f, -1f, +1f, -1f, -1f, +1f, +1f, +1f))
    private val cameraTexCoords: FloatBuffer = floatBuffer(FloatArray(8))
    private var program = 0
    private var textureLoc = -1

    /** External OES texture that ARCore writes the camera image into (pass to `Session.setCameraTextureName`). */
    var cameraTextureId = 0
        private set

    /** Allocates GL resources. Call from `GLSurfaceView.Renderer.onSurfaceCreated`. */
    fun createOnGlThread(assets: AssetManager) {
        val tex = IntArray(1)
        GLES30.glGenTextures(1, tex, 0)
        cameraTextureId = tex[0]
        val target = GLES11Ext.GL_TEXTURE_EXTERNAL_OES
        GLES30.glBindTexture(target, cameraTextureId)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(target, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

        val vs = compile(GLES30.GL_VERTEX_SHADER, assets.readText("shaders/background_show_camera.vert"))
        val fs = compile(GLES30.GL_FRAGMENT_SHADER, assets.readText("shaders/background_show_camera.frag"))
        program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vs)
        GLES30.glAttachShader(program, fs)
        GLES30.glLinkProgram(program)
        val ok = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, ok, 0)
        check(ok[0] != 0) { "background shader link failed: " + GLES30.glGetProgramInfoLog(program) }
        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(fs)
        textureLoc = GLES30.glGetUniformLocation(program, "u_CameraColorTexture")
    }

    /**
     * Re-queries the screen-quad UVs when the display geometry (rotation / view size) changed.
     * Must be called every frame before [draw].
     */
    fun updateDisplayGeometry(frame: Frame) {
        if (frame.hasDisplayGeometryChanged()) {
            ndcQuad.position(0); cameraTexCoords.position(0)
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, ndcQuad,
                Coordinates2d.TEXTURE_NORMALIZED, cameraTexCoords,
            )
        }
    }

    /** Draws the camera image over the whole viewport. */
    fun draw() {
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
        GLES30.glUniform1i(textureLoc, 0)
        ndcQuad.position(0); cameraTexCoords.position(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, ndcQuad)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 0, cameraTexCoords)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glDisableVertexAttribArray(0)
        GLES30.glDisableVertexAttribArray(1)
        GLES30.glDepthMask(true)
    }

    private fun compile(type: Int, src: String): Int {
        val s = GLES30.glCreateShader(type)
        GLES30.glShaderSource(s, src)
        GLES30.glCompileShader(s)
        val ok = IntArray(1)
        GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
        check(ok[0] != 0) { "shader compile failed: " + GLES30.glGetShaderInfoLog(s) }
        return s
    }

    private companion object {
        fun floatBuffer(v: FloatArray): FloatBuffer =
            ByteBuffer.allocateDirect(v.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(v); position(0) }

        fun AssetManager.readText(path: String): String = open(path).bufferedReader().use { it.readText() }
    }
}
