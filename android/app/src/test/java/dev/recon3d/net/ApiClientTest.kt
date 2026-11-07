package dev.recon3d.net

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import okhttp3.OkHttpClient
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.SocketPolicy
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
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
    /** Same client with the transfer-retry backoff disabled, for tests that exercise reconnects. */
    private lateinit var noSleep: ApiClient

    @Before fun setUp() {
        server = MockWebServer()
        server.start()
        val http = OkHttpClient.Builder()
            .connectTimeout(2, TimeUnit.SECONDS)
            .readTimeout(2, TimeUnit.SECONDS)
            // OkHttp's own silent retry would swallow the disconnects these tests script.
            .retryOnConnectionFailure(false)
            .build()
        // Same shape a user would type: scheme + host:port, no trailing slash.
        client = ApiClient("http://127.0.0.1:${server.port}", http)
        noSleep = ApiClient("http://127.0.0.1:${server.port}", http, sleep = {})
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

    // --- upload (resumable, docs/api.md "断点续传上传") ---

    private class RecordingListener : UploadListener {
        val created = mutableListOf<String>()
        val progress = mutableListOf<Float>()
        val retries = mutableListOf<Int>()
        override fun onCreated(uploadId: String) { created += uploadId }
        override fun onProgress(fraction: Float) { progress += fraction }
        override fun onRetry(attempt: Int) { retries += attempt }
    }

    private fun created(id: String = "abcdefabcdef") =
        MockResponse().setResponseCode(201).setBody("""{"upload_id":"$id","received":0}""")
    private fun received(n: Long) = MockResponse().setBody("""{"received":$n}""")
    private fun finished(id: String = "abcdefabcdef") =
        MockResponse().setResponseCode(201).setBody("""{"job_id":"$id"}""")
    private fun info(size: Long, received: Long, id: String = "abcdefabcdef") =
        MockResponse().setBody("""{"upload_id":"$id","size":$size,"received":$received}""")

    @Test fun uploadCreatesPutsChunksAndFinishes() = runTest {
        val zip = tmpZip(8 * 1024 * 1024 + 5)  // 2 full chunks + 5 bytes
        server.enqueue(created())
        server.enqueue(received(4194304)); server.enqueue(received(8388608)); server.enqueue(received(8388613))
        server.enqueue(finished())
        val l = RecordingListener()
        assertEquals("abcdefabcdef", client.upload(zip, null, "fast", l))
        val create = server.takeRequest()
        assertEquals("POST", create.method); assertEquals("/api/v1/uploads", create.path)
        val body = Json.parseToJsonElement(create.body.readUtf8()).jsonObject
        assertEquals(zip.length(), body["size"]!!.jsonPrimitive.long)
        assertEquals(sha256Hex(zip), body["sha256"]!!.jsonPrimitive.content)
        val put1 = server.takeRequest()
        assertEquals("PUT", put1.method); assertEquals("/api/v1/uploads/abcdefabcdef", put1.path)
        assertEquals("0", put1.getHeader("Upload-Offset")); assertEquals(4194304L, put1.bodySize)
        assertEquals("application/octet-stream", put1.getHeader("Content-Type"))
        assertEquals("4194304", server.takeRequest().getHeader("Upload-Offset"))
        val put3 = server.takeRequest()
        assertEquals("8388608", put3.getHeader("Upload-Offset")); assertEquals(5L, put3.bodySize)
        assertArrayEquals(zip.readBytes().takeLast(5).toByteArray(), put3.body.readByteArray())
        val finish = server.takeRequest()
        assertEquals("POST", finish.method); assertEquals("/api/v1/uploads/abcdefabcdef/finish", finish.path)
        assertEquals(listOf("abcdefabcdef"), l.created)
        assertEquals(1f, l.progress.last()); assertTrue(l.progress.zipWithNext().all { (x, y) -> y >= x })
        assertTrue(l.retries.isEmpty())
    }

    @Test fun uploadResumesFromServerReceivedAndCorrectsOffsetOn409() = runTest {
        val zip = tmpZip(6 * 1024 * 1024)
        server.enqueue(info(6291456, 1000))
        server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"偏移不匹配","received":2000}"""))
        server.enqueue(received(6291456))
        server.enqueue(finished())
        val l = RecordingListener()
        client.upload(zip, "abcdefabcdef", "fast", l)
        val get = server.takeRequest()
        assertEquals("GET", get.method); assertEquals("/api/v1/uploads/abcdefabcdef", get.path)
        assertEquals("1000", server.takeRequest().getHeader("Upload-Offset"))
        val put2 = server.takeRequest()
        assertEquals("2000", put2.getHeader("Upload-Offset")); assertEquals(4194304L, put2.bodySize)
        assertTrue(l.created.isEmpty()); assertTrue(l.retries.isEmpty())
    }

    @Test fun uploadRetriesAfterDisconnectThenContinues() = runTest {
        val zip = tmpZip(100)
        server.enqueue(created())
        server.enqueue(MockResponse().setSocketPolicy(SocketPolicy.DISCONNECT_DURING_REQUEST_BODY))
        server.enqueue(info(100, 40))
        server.enqueue(received(100))
        server.enqueue(finished())
        val l = RecordingListener()
        noSleep.upload(zip, null, "fast", l)
        assertEquals(listOf(1), l.retries)
        server.takeRequest(); server.takeRequest()
        assertEquals("GET", server.takeRequest().method)
        val put = server.takeRequest()
        assertEquals("40", put.getHeader("Upload-Offset")); assertEquals(60L, put.bodySize)
    }

    @Test fun uploadBusy409IsRetriedWithBackoff() = runTest {
        val zip = tmpZip(3)
        server.enqueue(created())
        server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"该上传正在接收中","received":0}"""))
        server.enqueue(info(3, 0))
        server.enqueue(received(3))
        server.enqueue(finished())
        val l = RecordingListener()
        noSleep.upload(zip, null, "fast", l)
        assertEquals(listOf(1), l.retries)
    }

    @Test fun uploadGivesUpAfterThirtyConsecutiveFailures() {
        val zip = tmpZip(10)
        server.enqueue(created())
        repeat(31) { server.enqueue(MockResponse().setSocketPolicy(SocketPolicy.DISCONNECT_AT_START)) }
        val l = RecordingListener()
        val e = apiError { noSleep.upload(zip, null, "fast", l) }
        assertEquals(ApiClient.MSG_UNREACHABLE, e.messageZh); assertEquals(30, l.retries.size)
    }

    @Test fun uploadFinishedBeforeKillReturnsExistingJob() = runTest {
        val zip = tmpZip(10)
        server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"上传不存在"}"""))
        server.enqueue(MockResponse().setBody("""{"job_id":"000000000000","state":"training","progress":0.1,"message":"训练中"}"""))
        assertEquals("000000000000", client.upload(zip, "000000000000", "fast", RecordingListener()))
        server.takeRequest()
        assertEquals("/api/v1/jobs/000000000000", server.takeRequest().path)
        assertEquals(2, server.requestCount)
    }

    @Test fun uploadRestartsWhenServerForgotTheUpload() = runTest {
        val zip = tmpZip(10)
        server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"上传不存在"}"""))
        server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"任务不存在"}"""))
        server.enqueue(created("111111111111"))
        server.enqueue(received(10))
        server.enqueue(finished("111111111111"))
        val l = RecordingListener()
        assertEquals("111111111111", client.upload(zip, "000000000000", "fast", l))
        assertEquals(listOf("111111111111"), l.created)
    }

    @Test fun uploadFatal4xxCarriesCodeWithoutRetry() {
        val zip = tmpZip(3)
        server.enqueue(created())
        server.enqueue(MockResponse().setResponseCode(413).setBody("""{"detail":"上传文件过大，上限 4 GB"}"""))
        val l = RecordingListener()
        val e = apiError { noSleep.upload(zip, null, "fast", l) }
        assertEquals("上传文件过大，上限 4 GB", e.messageZh); assertEquals(413, e.code)
        assertTrue(l.retries.isEmpty()); assertEquals(2, server.requestCount)
    }

    @Test fun uploadChecksumFailureCarriesServerDetailAndCode() {
        val zip = tmpZip(3)
        server.enqueue(created())
        server.enqueue(received(3))
        server.enqueue(MockResponse().setResponseCode(422).setBody("""{"detail":"文件校验失败，请重新上传"}"""))
        val e = apiError { client.upload(zip, null, "fast", RecordingListener()) }
        assertEquals("文件校验失败，请重新上传", e.messageZh); assertEquals(422, e.code)
    }

    @Test fun uploadTooLargeUsesServerDetail() {
        server.enqueue(MockResponse().setResponseCode(413).setBody("""{"detail":"上传文件过大，上限 4 GB"}"""))
        assertEquals("上传文件过大，上限 4 GB", apiError { client.upload(tmpZip(16), null, "fast", RecordingListener()) }.messageZh)
    }

    @Test fun uploadValidationErrorWithListDetailGivesCode() {
        // FastAPI 422 bodies carry a list in `detail`, not a Chinese string.
        server.enqueue(MockResponse().setResponseCode(422).setBody(
            """{"detail":[{"loc":["body","size"],"msg":"Field required","type":"missing"}]}"""
        ))
        assertEquals("计算端返回错误：HTTP 422", apiError { client.upload(tmpZip(16), null, "fast", RecordingListener()) }.messageZh)
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

    @Test fun downloadResumesPartWithRangeAnd206() = runTest {
        val dest = File(tmp.root, "s/scene.splat"); dest.parentFile!!.mkdirs()
        File(dest.path + ".part").writeBytes(byteArrayOf(1, 2))
        server.enqueue(MockResponse().setResponseCode(206).setHeader("Content-Range", "bytes 2-3/4")
            .setBody(Buffer().write(byteArrayOf(3, 4))))
        client.downloadScene("j", dest)
        assertEquals("bytes=2-", server.takeRequest().getHeader("Range"))
        assertArrayEquals(byteArrayOf(1, 2, 3, 4), dest.readBytes())
        assertFalse(File(dest.path + ".part").exists())
    }

    @Test fun downloadPlain200OverwritesPart() = runTest {
        val dest = File(tmp.root, "s/scene.splat"); dest.parentFile!!.mkdirs()
        File(dest.path + ".part").writeBytes(byteArrayOf(9, 9))
        server.enqueue(MockResponse().setBody(Buffer().write(byteArrayOf(1, 2, 3))))
        client.downloadScene("j", dest)
        assertArrayEquals(byteArrayOf(1, 2, 3), dest.readBytes())
    }

    @Test fun downloadRange416RestartsFromScratch() = runTest {
        val dest = File(tmp.root, "s/scene.splat"); dest.parentFile!!.mkdirs()
        File(dest.path + ".part").writeBytes(byteArrayOf(9, 9, 9, 9, 9))
        server.enqueue(MockResponse().setResponseCode(416))
        server.enqueue(MockResponse().setBody(Buffer().write(byteArrayOf(1, 2))))
        client.downloadScene("j", dest)
        assertEquals("bytes=5-", server.takeRequest().getHeader("Range"))
        assertNull(server.takeRequest().getHeader("Range"))
        assertArrayEquals(byteArrayOf(1, 2), dest.readBytes())
    }

    @Test fun downloadRetriesAfterDropAndResumes() = runTest {
        val dest = File(tmp.root, "s/scene.splat")
        server.enqueue(MockResponse().setBody(Buffer().write(byteArrayOf(1, 2, 3, 4)))
            .setSocketPolicy(SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY))
        server.enqueue(MockResponse().setResponseCode(206).setHeader("Content-Range", "bytes 2-3/4")
            .setBody(Buffer().write(byteArrayOf(3, 4))))
        val retries = mutableListOf<Int>()
        noSleep.downloadScene("j", dest) { retries += it }
        assertEquals(listOf(1, 0), retries)  // 0 = the retried request got through
        assertNull(server.takeRequest().getHeader("Range"))
        assertEquals("bytes=2-", server.takeRequest().getHeader("Range"))
        assertArrayEquals(byteArrayOf(1, 2, 3, 4), dest.readBytes())
    }

    @Test fun downloadSceneNotDoneUsesServerDetailAndLeavesNoFile() {
        server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"任务尚未完成"}"""))
        val dest = File(tmp.root, "scene.splat")
        assertEquals("任务尚未完成", apiError { client.downloadScene("abc", dest) }.messageZh)
        assertFalse(dest.exists())
    }

    // --- deleteJob (docs/superpowers/specs/2026-11-05-recon3d-delete-design.md) ---

    @Test fun deleteJobReturnsOn204And404() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(204))
        server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"任务不存在"}"""))
        client.deleteJob("abc123abc123")
        client.deleteJob("abc123abc123")
        val req = server.takeRequest()
        assertEquals("DELETE", req.method)
        assertEquals("/api/v1/jobs/abc123abc123", req.path)
    }

    @Test fun deleteJob409SurfacesDetailAndCode() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"任务正在训练，完成后再删"}"""))
        val e = assertThrows(ApiException::class.java) { runBlocking { client.deleteJob("abc123abc123") } }
        assertEquals("任务正在训练，完成后再删", e.messageZh)
        assertEquals(409, e.code)
    }

    @Test fun deleteJobTransportErrorIsUnreachable() = runBlocking {
        server.shutdown()
        val e = assertThrows(ApiException::class.java) { runBlocking { client.deleteJob("abc123abc123") } }
        assertEquals(ApiClient.MSG_UNREACHABLE, e.messageZh)
    }

    // --- mode / rerun ---

    @Test fun uploadSendsModeInCreateBody() = runTest {
        val zip = tmpZip(10)
        server.enqueue(created()); server.enqueue(received(10)); server.enqueue(finished())
        client.upload(zip, null, "precise", RecordingListener())
        val body = Json.parseToJsonElement(server.takeRequest().body.readUtf8()).jsonObject
        assertEquals("precise", body["mode"]!!.jsonPrimitive.content)
    }

    @Test fun rerunJobPostsModeAndReturnsNewId() = runTest {
        server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"bbbbbbbbbbbb"}"""))
        assertEquals("bbbbbbbbbbbb", client.rerunJob("aaaaaaaaaaaa", "precise"))
        val req = server.takeRequest()
        assertEquals("POST", req.method); assertEquals("/api/v1/jobs/aaaaaaaaaaaa/rerun", req.path)
        assertEquals("precise", Json.parseToJsonElement(req.body.readUtf8()).jsonObject["mode"]!!.jsonPrimitive.content)
    }

    @Test fun rerunJobErrorsCarryServerDetailAndCode() {
        server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"任务不存在"}"""))
        val e = apiError { client.rerunJob("aaaaaaaaaaaa", "fast") }
        assertEquals("任务不存在", e.messageZh); assertEquals(404, e.code)
        server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"上传未完成"}"""))
        assertEquals(409, apiError { client.rerunJob("aaaaaaaaaaaa", "fast") }.code)
    }

    // --- objects (phase 2) ---

    @Test fun requestObjectsPostsAndDecodesTheObjectsBlock() = runTest {
        server.enqueue(MockResponse().setResponseCode(202).setBody("""{"job_id":"aaaaaaaaaaaa","objects":{"state":"queued","progress":0.0,"message":"排队中","count":0}}"""))
        val o = client.requestObjects("aaaaaaaaaaaa")
        assertEquals("queued", o.state)
        val req = server.takeRequest()
        assertEquals("POST", req.method); assertEquals("/api/v1/jobs/aaaaaaaaaaaa/objects", req.path)
    }

    @Test fun requestObjectsErrorsCarryServerDetail() {
        server.enqueue(MockResponse().setResponseCode(409).setBody("""{"detail":"正在识别"}"""))
        val e = apiError { client.requestObjects("aaaaaaaaaaaa") }
        assertEquals("正在识别", e.messageZh); assertEquals(409, e.code)
    }

    @Test fun jobStatusWithoutObjectsBlockDecodesToNone() = runTest {
        server.enqueue(MockResponse().setBody("""{"job_id":"j","state":"done","progress":1.0,"message":"完成","mode":"fast"}"""))
        assertEquals(ObjectsStatus(), client.job("j").objects)
        server.enqueue(MockResponse().setBody("""{"job_id":"j","state":"done","progress":1.0,"message":"完成","objects":{"state":"running","progress":0.4,"message":"识别物体 (检测)","count":0}}"""))
        assertEquals(ObjectsStatus("running", 0.4f, "识别物体 (检测)", 0), client.job("j").objects)
    }

    @Test fun downloadObjectsWritesTheZip() = runTest {
        server.enqueue(MockResponse().setBody(Buffer().write(byteArrayOf(80, 75, 5, 6))))
        val dest = File(tmp.root, "x/objects.zip")
        client.downloadObjects("aaaaaaaaaaaa", dest)
        assertArrayEquals(byteArrayOf(80, 75, 5, 6), dest.readBytes())
        assertEquals("/api/v1/jobs/aaaaaaaaaaaa/objects.zip", server.takeRequest().path)
    }

    @Test fun downloadObjectsNotFoundSurfacesDetail() {
        server.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail":"没有物体表"}"""))
        val e = apiError { client.downloadObjects("aaaaaaaaaaaa", File(tmp.root, "objects.zip")) }
        assertEquals("没有物体表", e.messageZh); assertEquals(404, e.code)
    }
}
