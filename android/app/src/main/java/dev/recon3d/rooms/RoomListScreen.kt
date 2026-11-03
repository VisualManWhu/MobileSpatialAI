package dev.recon3d.rooms

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.recon3d.upload.CaptureRow
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/** MainActivity's outer Scaffold already pads for the system bars. */
private val NoInsets = WindowInsets(0, 0, 0, 0)

private val dateFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")

fun formatCreatedAt(epochMillis: Long, zone: ZoneId = ZoneId.systemDefault()): String =
    dateFormat.format(Instant.ofEpochMilli(epochMillis).atZone(zone))

/** Start screen: all rooms, "新建房间", gear → settings. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RoomListScreen(
    repo: RoomRepository,
    onOpenRoom: (roomId: String) -> Unit,
    onOpenSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val rooms by repo.rooms().collectAsState()
    var creating by rememberSaveable { mutableStateOf(false) }

    Scaffold(
        modifier = modifier,
        contentWindowInsets = NoInsets,
        topBar = {
            TopAppBar(
                windowInsets = NoInsets,
                title = { Text("我的房间") },
                actions = {
                    IconButton(onClick = onOpenSettings, modifier = Modifier.semantics { contentDescription = "设置" }) {
                        Text("⚙", fontSize = 22.sp)
                    }
                },
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(onClick = { creating = true }) { Text("新建房间") }
        },
    ) { padding ->
        if (rooms.isEmpty()) {
            Column(Modifier.padding(padding).padding(24.dp)) {
                Text("还没有房间，点右下角“新建房间”开始。", style = MaterialTheme.typography.bodyLarge)
            }
        } else {
            LazyColumn(
                Modifier.fillMaxSize().padding(padding),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                items(rooms, key = { it.id }) { room ->
                    Card(Modifier.fillMaxWidth().clickable { onOpenRoom(room.id) }) {
                        Column(Modifier.padding(16.dp)) {
                            Text(room.name, style = MaterialTheme.typography.titleMedium)
                            Text(
                                "创建于 ${formatCreatedAt(room.createdAt)} · ${room.captures.size} 次拍摄",
                                style = MaterialTheme.typography.bodyMedium,
                            )
                        }
                    }
                }
            }
        }
    }

    if (creating) {
        NewRoomDialog(
            onDismiss = { creating = false },
            onCreate = { name ->
                creating = false
                onOpenRoom(repo.createRoom(name).id)
            },
        )
    }
}

@Composable
private fun NewRoomDialog(onDismiss: () -> Unit, onCreate: (String) -> Unit) {
    var name by rememberSaveable { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("新建房间") },
        text = {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                label = { Text("房间名称") },
                placeholder = { Text("例如 卧室") },
                singleLine = true,
            )
        },
        confirmButton = { TextButton(enabled = name.isNotBlank(), onClick = { onCreate(name) }) { Text("创建") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
    )
}

/** One room: its captures with status/progress and actions, plus "拍摄". */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RoomScreen(
    repo: RoomRepository,
    roomId: String,
    onCapture: () -> Unit,
    onUpload: (captureId: String) -> Unit,
    onRetry: (captureId: String) -> Unit,
    onView: (captureId: String) -> Unit,
    onViewPoints: (captureId: String) -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val rooms by repo.rooms().collectAsState()
    val room = rooms.firstOrNull { it.id == roomId } ?: return

    Scaffold(
        modifier = modifier,
        contentWindowInsets = NoInsets,
        topBar = {
            TopAppBar(
                windowInsets = NoInsets,
                title = { Text(room.name) },
                navigationIcon = {
                    IconButton(onClick = onBack, modifier = Modifier.semantics { contentDescription = "返回" }) {
                        Text("←", fontSize = 22.sp)
                    }
                },
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(onClick = onCapture) { Text("拍摄") }
        },
    ) { padding ->
        if (room.captures.isEmpty()) {
            Column(Modifier.padding(padding).padding(24.dp)) {
                Text("还没有拍摄，点右下角“拍摄”开始。", style = MaterialTheme.typography.bodyLarge)
            }
        } else {
            LazyColumn(
                Modifier.fillMaxSize().padding(padding),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                items(room.captures.asReversed(), key = { it.id }) { rec ->
                    CaptureRow(
                        record = rec,
                        onUpload = { onUpload(rec.id) },
                        onRetry = { onRetry(rec.id) },
                        onView = { onView(rec.id) },
                        onViewPoints = { onViewPoints(rec.id) },
                    )
                }
            }
        }
    }
}
