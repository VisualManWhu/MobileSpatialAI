package dev.recon3d.net

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.random.Random

class ApiClientTest {
    @get:Rule val tmp = TemporaryFolder()

    private lateinit var server: MockWebServer
    private lateinit var client: ApiClient

    @Before fun setUp() {
        server = MockWebServer()
        server.start()
        val http = OkHttpClient.Builder()
            .connectTimeout(2, TimeUnit.SECONDS)
            .readTimeout(2, TimeUnit.SECONDS)
            .build()
        // Same shape a user would type: scheme + host:port, no trailing slash.
        client = ApiClient("http://127.0.0.1:${server.port}", http)
    }

    @After fun tearDown() {
        server.shutdown()
    }

    private fun tmpZip(size: Int): File =
        tmp.newFile("capture.zip").apply { writeBytes(Random(42).nextBytes(size)) }

    private fun apiError(block: suspend () -> Unit): ApiException =
        assertThrows(ApiException::class.java) { runBlocking { block() } }

    // --- health ---

    @Test fun healthParses() = runTest {
        server.enqueue(MockResponse().setBody("""{"version":"0.1","gpu":"RTX 4060","profile":"laptop"}"""))
        assertEquals("laptop", client.health().profile)
        assertEquals("/api/v1/health", server.takeRequest().path)
    }

    @Test fun healthParsesAllFieldsAndIgnoresUnknown() = runTest {
        server.enqueue(MockResponse().setBody(
            """{"version":"0.1.0","gpu":"NVIDIA GeForce RTX 4060","profile":"server","extra":1}"""
        ))
        assertEquals(Health("0.1.0", "NVIDIA GeForce RTX 4060", "server"), client.health())
    }

    @Test fun connectionRefusedGivesChineseMessage() = runTest {
        server.shutdown()
        val e = assertThrows(ApiException::class.java) { runBlocking { client.health() } }
        assertTrue(e.messageZh.contains("连不上计算端"))
    }

    @Test fun unknownHostGivesConnectMessage() {
        val c = ApiClient("http://no-such-host.invalid:8765", OkHttpClient())
        assertEquals("连不上计算端，请检查 Wi-Fi 和地址", apiError { c.health() }.messageZh)
    }

    @Test fun readTimeoutGivesConnectMessage() {
        server.enqueue(MockResponse().setBody("{}").setHeadersDelay(1, TimeUnit.SECONDS))
        val http = OkHttpClient.Builder()
            .readTimeout(200, TimeUnit.MILLISECONDS)
            .retryOnConnectionFailure(false)
            .build()
        val c = ApiClient("http://127.0.0.1:${server.port}", http)
        assertEquals("连不上计算端，请检查 Wi-Fi 和地址", apiError { c.health() }.messageZh)
    }

    @Test fun malformedJsonGivesParseMessage() {
        server.enqueue(MockResponse().setBody("<html>not json</html>"))
        assertEquals("计算端返回的数据无法解析", apiError { client.health() }.messageZh)
    }

    @Test fun missingFieldGivesParseMessage() {
        server.enqueue(MockResponse().setBody("""{"version":"0.1"}"""))
        assertEquals("计算端返回的数据无法解析", apiError { client.health() }.messageZh)
    }

    @Test fun httpErrorWithoutDetailGivesCode() {
        server.enqueue(MockResponse().setResponseCode(500).setBody("Internal Server Error"))
        assertEquals("计算端返回错误：HTTP 500", apiError { client.health() }.messageZh)
    }

    // --- upload ---

    @Test fun uploadReturnsJobId() = runTest {
        server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"abc"}"""))
        assertEquals("abc", client.upload(tmpZip(1024)) {})
        assertTrue(server.takeRequest().body.readUtf8().contains("name=\"capture\""))
    }

    @Test fun uploadSendsMultipartFileToJobs() = runTest {
        val zip = tmpZip(4096)
        server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"3f9c0a1b2d4e"}"""))
        client.upload(zip) {}
        val req = server.takeRequest()
        assertEquals("POST", req.method)
        assertEquals("/api/v1/jobs", req.path)
        assertTrue(req.getHeader("Content-Type")!!.startsWith("multipart/form-data"))
        val body = req.body.readByteArray()
        val bodyText = String(body, Charsets.ISO_8859_1)
        assertTrue(bodyText.contains("filename=\"capture.zip\""))
        assertTrue(Buffer().write(body).indexOf(okio.ByteString.of(*zip.readBytes())) >= 0)
    }

    @Test fun uploadReportsMonotonicProgressEndingAtOne() = runTest {
        server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"abc"}"""))
        val seen = mutableListOf<Float>()
        client.upload(tmpZip(300_000)) { seen += it }
        assertTrue("expected several progress updates, got $seen", seen.size > 1)
        assertEquals(1f, seen.last(), 0.0001f)
        assertTrue(seen.zipWithNext().all { (a, b) -> b >= a })
        assertTrue(seen.all { it in 0f..1f })
    }

    @Test fun uploadTooLargeUsesServerDetail() {
        server.enqueue(MockResponse().setResponseCode(413).setBody("""{"detail":"上传文件过大，上限 4 GB"}"""))
        assertEquals("上传文件过大，上限 4 GB", apiError { client.upload(tmpZip(16)) {} }.messageZh)
    }

    @Test fun uploadValidationErrorWithListDetailGivesCode() {
        // FastAPI 422 bodies carry a list in `detail`, not a Chinese string.
        server.enqueue(MockResponse().setResponseCode(422).setBody(
            """{"detail":[{"loc":["body","capture"],"msg":"Field required","type":"missing"}]}"""
        ))
        assertEquals("计算端返回错误：HTTP 422", apiError { client.upload(tmpZip(16)) {} }.messageZh)
    }

    // --- job ---

    @Test fun jobParsesStatus() = runTest {
        server.enqueue(MockResponse().setBody(
            """{"job_id":"3f9c0a1b2d4e","state":"training","progress":0.42,"message":"训练中"}"""
        ))
        val s = client.job("3f9c0a1b2d4e")
        assertEquals(JobStatus("3f9c0a1b2d4e", JobState.TRAINING, 0.42f, "训练中"), s)
        assertEquals("/api/v1/jobs/3f9c0a1b2d4e", server.takeRequest().path)
    }

    @Test fun jobParsesEveryState() = runTest {
        val states = mapOf(
            "queued" to JobState.QUEUED, "training" to JobState.TRAINING,
            "exporting" to JobState.EXPORTING, "done" to JobState.DONE, "failed" to JobState.FAILED,
        )
        for ((wire, expected) in states) {
            server.enqueue(MockResponse().setBody(
                """{"job_id":"a","state":"$wire","progress":1,"message":"m"}"""
            ))
            assertEquals(expected, client.job("a").state)
        }
    }

    @Test fun jobNotFoundUsesServerDetail() {
        server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"任务不存在"}"""))
        assertEquals("任务不存在", apiError { client.job("nope") }.messageZh)
    }

    // --- downloadScene ---

    @Test fun downloadSceneWritesBytes() = runTest {
        val bytes = Random(7).nextBytes(100_000)
        server.enqueue(MockResponse()
            .setHeader("Content-Type", "application/octet-stream")
            .setBody(Buffer().write(bytes)))
        val dest = File(tmp.root, "out/scene.splat")
        client.downloadScene("abc", dest)
        assertArrayEquals(bytes, dest.readBytes())
        assertEquals("/api/v1/jobs/abc/scene.splat", server.takeRequest().path)
        assertFalse(File(dest.path + ".part").exists())
    }

    @Test fun downloadSceneNotDoneUsesServerDetailAndLeavesNoFile() {
        server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"任务尚未完成"}"""))
        val dest = File(tmp.root, "scene.splat")
        assertEquals("任务尚未完成", apiError { client.downloadScene("abc", dest) }.messageZh)
        assertFalse(dest.exists())
    }
}
