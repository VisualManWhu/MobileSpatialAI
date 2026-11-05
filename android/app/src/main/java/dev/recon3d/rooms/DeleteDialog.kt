package dev.recon3d.rooms

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/** One button of the delete dialog; [destructive] ones are drawn in the error colour. */
data class DeleteChoice(val mode: DeleteMode, val label: String, val destructive: Boolean)

fun captureDeleteTitle(record: CaptureRecord): String = "删除「${record.id}」"

fun captureDeleteBody(options: List<DeleteMode>): String = when (options) {
    listOf(DeleteMode.LOCAL) -> "删除手机上的拍摄包，不可恢复。"
    listOf(DeleteMode.CLOUD) -> "手机和计算端上的拍摄包与模型都删掉，不可恢复。"
    else -> "彻底删除：手机和计算端上的拍摄包与模型都删掉，不可恢复。\n仅删手机文件：释放手机空间，保留记录，之后可从计算端重新下载模型。"
}

fun captureDeleteChoices(options: List<DeleteMode>): List<DeleteChoice> = options.map { mode ->
    when (mode) {
        DeleteMode.CLOUD -> DeleteChoice(mode, "彻底删除", destructive = true)
        // A never-uploaded capture has only this choice, and then it is plain "删除" (and destructive).
        DeleteMode.LOCAL -> if (options.size == 1) DeleteChoice(mode, "删除", destructive = true)
        else DeleteChoice(mode, "仅删手机文件", destructive = false)
    }
}

fun roomDeleteTitle(room: Room): String = "删除「${room.name}」"

fun roomDeleteBody(room: Room, options: List<DeleteMode>): String =
    if (options == listOf(DeleteMode.LOCAL)) "删除手机上的这个房间，不可恢复。"
    else "彻底删除：这个房间的 ${room.captures.size} 条拍摄在手机和计算端都删掉，不可恢复。\n仅删手机：删掉手机上的房间和文件，计算端的任务保留。"

fun roomDeleteChoices(options: List<DeleteMode>): List<DeleteChoice> = options.map { mode ->
    when (mode) {
        DeleteMode.CLOUD -> DeleteChoice(mode, "彻底删除", destructive = true)
        DeleteMode.LOCAL -> if (options.size == 1) DeleteChoice(mode, "删除", destructive = true)
        else DeleteChoice(mode, "仅删手机", destructive = false)
    }
}

/** Confirmation dialog: one button per [choices] plus 取消; disabled with a progress bar while [busy]. */
@Composable
fun DeleteDialog(
    title: String,
    body: String,
    choices: List<DeleteChoice>,
    busy: Boolean,
    onChoose: (DeleteMode) -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = { if (!busy) onDismiss() },
        title = { Text(title) },
        text = {
            Column {
                Text(body)
                if (busy) LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 12.dp))
            }
        },
        confirmButton = {
            // Material puts confirmButton after dismissButton; keep the destructive choice right-most.
            for (c in choices) {
                TextButton(
                    onClick = { onChoose(c.mode) },
                    enabled = !busy,
                    colors = if (c.destructive) ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)
                    else ButtonDefaults.textButtonColors(),
                ) { Text(c.label) }
            }
        },
        dismissButton = { TextButton(onClick = onDismiss, enabled = !busy) { Text("取消") } },
    )
}
