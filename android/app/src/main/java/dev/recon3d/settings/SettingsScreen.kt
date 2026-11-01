package dev.recon3d.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import dev.recon3d.net.Api
import dev.recon3d.net.ApiException
import dev.recon3d.net.Health
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

/** Server address field + "测试连接". [apiFor] builds an [Api] for a base URL (fakeable). */
@Composable
fun SettingsScreen(
    settings: ServerSettings,
    apiFor: (String) -> Api,
    modifier: Modifier = Modifier,
    onOpenCapture: (() -> Unit)? = null,
) {
    var address by rememberSaveable { mutableStateOf("") }
    var result by remember { mutableStateOf<String?>(null) }
    var testing by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    LaunchedEffect(Unit) {
        if (address.isEmpty()) address = settings.baseUrl.first().orEmpty()
    }

    Column(
        modifier = modifier.padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("计算端设置", style = MaterialTheme.typography.headlineSmall)
        OutlinedTextField(
            value = address,
            onValueChange = { address = it },
            label = { Text("计算端地址") },
            placeholder = { Text("例如 192.168.1.20:8765 或 100.x.x.x:8765") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri, imeAction = ImeAction.Done),
            modifier = Modifier.fillMaxWidth(),
        )
        Button(
            enabled = !testing,
            onClick = {
                if (address.isBlank()) {
                    result = "请输入计算端地址"
                    return@Button
                }
                val url = normalizeBaseUrl(address)
                address = url
                testing = true
                result = "正在连接…"
                scope.launch {
                    result = try {
                        settings.setBaseUrl(url)
                        connectedMessage(apiFor(url).health())
                    } catch (e: ApiException) {
                        e.messageZh
                    } finally {
                        testing = false
                    }
                }
            },
        ) { Text("测试连接") }
        result?.let { Text(it, style = MaterialTheme.typography.bodyLarge) }
        // TEMP until Task 12 adds the rooms list.
        onOpenCapture?.let { OutlinedButton(onClick = it) { Text("（临时）开始拍摄") } }
    }
}

/** e.g. `已连接：RTX 4060（笔记本档）` for `NVIDIA GeForce RTX 4060` / `laptop`. */
fun connectedMessage(health: Health): String {
    val gpu = health.gpu.removePrefix("NVIDIA ").removePrefix("GeForce ")
    val profile = when (health.profile) {
        "laptop" -> "笔记本档"
        "server" -> "服务器档"
        else -> health.profile
    }
    return "已连接：$gpu（$profile）"
}
