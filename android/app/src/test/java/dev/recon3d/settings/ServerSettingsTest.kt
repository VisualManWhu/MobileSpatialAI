package dev.recon3d.settings

import dev.recon3d.net.Health
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ServerSettingsTest {
    @Test fun prependsHttpForBareLanAddress() {
        assertEquals("http://192.168.1.20:8765", normalizeBaseUrl("192.168.1.20:8765"))
    }

    @Test fun acceptsTailscaleAddress() {
        assertEquals("http://100.101.102.103:8765", normalizeBaseUrl("100.101.102.103:8765"))
    }

    @Test fun trimsWhitespaceAndTrailingSlashes() {
        assertEquals("http://192.168.1.20:8765", normalizeBaseUrl("  192.168.1.20:8765//  "))
    }

    @Test fun keepsExistingScheme() {
        assertEquals("http://laptop.local:8765", normalizeBaseUrl("http://laptop.local:8765/"))
        assertEquals("HTTP://10.0.0.2:8765", normalizeBaseUrl("HTTP://10.0.0.2:8765"))
    }

    @Test fun defaultsPortWhenMissing() {
        assertEquals("http://192.168.1.20:8765", normalizeBaseUrl("192.168.1.20"))
        assertEquals("http://100.101.102.103:8765", normalizeBaseUrl("100.101.102.103/"))
        assertEquals("https://example.org:8765", normalizeBaseUrl("https://example.org"))
        assertEquals("http://laptop.local:8765/base", normalizeBaseUrl("laptop.local/base"))
        assertEquals("http://[fd7a::1]:8765", normalizeBaseUrl("http://[fd7a::1]"))
    }

    @Test fun keepsExplicitPort() {
        assertEquals("http://192.168.1.20:9000", normalizeBaseUrl("192.168.1.20:9000"))
        assertEquals("https://example.org:443", normalizeBaseUrl("https://example.org:443"))
        assertEquals("http://[fd7a::1]:9000", normalizeBaseUrl("[fd7a::1]:9000"))
    }

    @Test fun rejectsBlank() {
        assertThrows(IllegalArgumentException::class.java) { normalizeBaseUrl("   ") }
    }

    @Test fun connectedMessageForLaptop() {
        assertEquals(
            "已连接：RTX 4060（笔记本档）",
            connectedMessage(Health("0.1.0", "NVIDIA GeForce RTX 4060", "laptop")),
        )
    }

    @Test fun connectedMessageForServer() {
        assertEquals(
            "已连接：RTX 4090（服务器档）",
            connectedMessage(Health("0.1.0", "NVIDIA GeForce RTX 4090", "server")),
        )
    }

    @Test fun connectedMessageKeepsUnknownProfileAndPlainGpuName() {
        assertEquals("已连接：A100（other）", connectedMessage(Health("0.1.0", "A100", "other")))
    }
}
