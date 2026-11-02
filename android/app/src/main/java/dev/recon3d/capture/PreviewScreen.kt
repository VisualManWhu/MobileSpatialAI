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
import dev.recon3d.viewer.SceneViewer

/**
 * Post-capture check: the ARCore feature points accumulated while recording, as a turnable cloud, so the
 * user can see whether the room's shape came out before spending minutes on an upload.
 */
@Composable
fun PreviewScreen(record: CaptureRecord, onUpload: () -> Unit, onRetake: () -> Unit, modifier: Modifier = Modifier) {
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
            Button(onClick = onUpload) { Text("上传") }
        }
    }
}
