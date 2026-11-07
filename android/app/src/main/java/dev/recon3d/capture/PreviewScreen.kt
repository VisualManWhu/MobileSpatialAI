package dev.recon3d.capture

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import dev.recon3d.rooms.CaptureRecord
import dev.recon3d.rooms.MODE_FAST
import dev.recon3d.rooms.MODE_PRECISE
import dev.recon3d.upload.captureStatusText
import dev.recon3d.viewer.SceneViewer

/**
 * The capture-time point cloud (keyframe depth back-projected while recording) as a turnable cloud.
 *
 * Right after recording ([onUpload] and [onRetake] given) it is the check before spending minutes on an upload;
 * [onUpload] gets the chosen reconstruction mode ([MODE_FAST] | [MODE_PRECISE]). Opened from the room while the
 * capture uploads or trains (both null) it is read-only: the bottom shows the live status line instead of the buttons.
 */
@Composable
fun PreviewScreen(
    record: CaptureRecord,
    onUpload: ((mode: String) -> Unit)?,
    onRetake: (() -> Unit)?,
    modifier: Modifier = Modifier,
) {
    Column(modifier.fillMaxSize().background(Color(0xFF111111))) {
        Text(
            "拍摄预览 · ${record.pointCount} 个点 · ${record.frameCount} 张照片",
            color = Color.White,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth().padding(12.dp),
        )
        Box(Modifier.weight(1f).fillMaxWidth()) {
            val path = record.pointsPath
            if (path != null) {
                SceneViewer(path)
            } else {
                Text(
                    "这次拍摄没有生成点云（没有深度数据，或保存失败）",
                    color = Color.White, textAlign = TextAlign.Center,
                    modifier = Modifier.align(Alignment.Center).padding(24.dp),
                )
            }
        }
        if (onUpload != null && onRetake != null) {
            Text(
                "看一眼房间的大致形状是否完整；有大块空白就重拍。",
                color = Color.White.copy(alpha = 0.7f), textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
            )
            Row(
                Modifier.fillMaxWidth().padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterHorizontally),
            ) {
                OutlinedButton(onClick = onRetake, colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White)) {
                    Text("重拍")
                }
                OutlinedButton(
                    onClick = { onUpload(MODE_FAST) },
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White),
                ) { Text("快速上传") }
                Button(onClick = { onUpload(MODE_PRECISE) }) { Text("高精上传") }
            }
        } else {
            Text(
                captureStatusText(record),
                color = Color.White, textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(16.dp),
            )
        }
    }
}
