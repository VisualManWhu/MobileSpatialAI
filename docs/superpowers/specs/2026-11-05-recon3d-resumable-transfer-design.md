# Recon3D 断点续传设计（上传 + 下载）

日期：2026-11-05。状态：待实现。

## 1. 为什么

真机上传实测（`docs/acceptance/phase1.md` §5 问题 2）：家用路由器下手机 → 笔记本吞吐 < 40 KB/s，71 MB 拍摄包十次都在几 KB ~ 2 MB 处断（`Software caused connection abort` / 服务端 `ClientDisconnect`）。现在的 `POST /jobs` 是一次性 multipart：断一次就从零重传，永远传不完。用户的原话：烦的是**老断**，不是慢。

连通性这一层已经解决、不需要改代码：笔记本与手机都装了 Tailscale（`100.81.205.31` ↔ `100.118.71.73`，同一 Wi-Fi 下走局域网直连 24 ms，外面走流量时走它的通道），从手机经 Tailscale 地址请求 `/api/v1/health` 返回 200，Windows 把 Tailscale 网卡归为"专用网络"，现有防火墙规则已覆盖。App 里只要填 `100.81.205.31`。

本设计只做一件事：**断了接着传，不从头来**。上传和下载都是。

## 2. 目标 / 非目标

目标：
- 上传中断（任何原因：Wi-Fi 掉线、切到流量、服务端短暂不可达）后，App 自动从服务端已收到的字节处继续，用户看到的是"上传中 37% · 网络中断，正在重连 (3)"，不是"失败"。
- App 进程被杀后再打开，同一次上传照样接着传（上传 id 持久化在 `rooms.json`）。
- 下载 `scene.splat`（20–45 MB）同样续传。
- 服务端对未完成的上传有上限和过期清理，不会被半截文件撑满。
- 完整性：服务端收齐后校验 SHA-256，错一个字节就拒绝。

非目标：
- 锁屏 / 切后台时继续传：那要前台服务，已在另一分支上做，这次不合并；传的时候仍需 App 在前台（上传页已保持常亮）。
- 让慢的链路变快（并发分块、压缩）。不做。
- 旧的一次性 `POST /jobs`：保留不动（测试与命令行脚本还在用），文档标为旧接口；手机改走新接口。

反目标（做成这样算失败）：
- 服务端重启后手机"重试"却从零开始——上传目录必须落盘、重启后可继续。
- 两个并发请求把同一个 part 文件写乱——同一上传同一时刻只允许一个写入。
- 服务端挂了 App 无限转圈没有出口——连续 30 次（约 15 分钟）仍不可达才标失败，且"重试"按钮从断点继续而不是重传。

## 3. 协议（新增，前缀 `/api/v1`，写入 `docs/api.md`）

与 tus 思路相同、不引入 tus 库。所有错误响应体仍是 `{"detail": "<中文>"}`。

### POST /uploads

请求 JSON：`{"size": 123456789, "sha256": "<64 位十六进制>"}`。
响应 `201`：`{"upload_id": "<12 位十六进制>", "received": 0}`。
- `size` 超过 `--max-upload-gb` 上限：`413`，detail 同旧接口（"上传文件过大，上限 4 GB"）。
- `size` ≤ 0 或 `sha256` 格式不对：`422`。

### PUT /uploads/{id}

请求头 `Upload-Offset: <已收字节数>`，请求体 `application/octet-stream`，任意长度（App 用 4 MB）。服务端**边收边写**：连接中途断了，已经写进磁盘的字节保留，`received` 相应前进——所以块大小不影响"断了丢多少"，丢的只是还没写盘的网络缓冲。
响应 `200`：`{"received": <写完后的总字节数>}`。
- `Upload-Offset` 与服务端 `received` 不一致：`409`，`{"detail":"偏移不匹配","received": n}`。App 据此纠正偏移重发。
- 写入后超过声明的 `size`：`413`，并把这次上传整个删掉（声明和实际不符，说明客户端状态错了）。
- 同一上传已有一个 PUT 在处理：`409`，`{"detail":"该上传正在接收中","received": n}`。App 把它当作一次普通失败（计入重试次数、退避后重问 `GET`），不紧循环。半开 TCP 连接可能让旧 PUT 卡住几分钟，所以服务端 PUT 有 **30 s 空闲超时**：连续 30 s 收不到请求体就当断开处理（已写入的字节保留、释放锁），旧连接最多占锁 30 s。
- 已经 finish 过的上传：`409`，`{"detail":"上传已完成"}`。
- 不存在：`404`，`{"detail":"上传不存在"}`。

### GET /uploads/{id}

响应 `200`：`{"upload_id": "...", "size": N, "received": n}`。App 在每次重连后先问这个，再从 `received` 继续。不存在：`404`。

### POST /uploads/{id}/finish

`received == size` 且 SHA-256 一致 → 创建任务、入队，响应 `201`：`{"job_id": "<同 upload_id>"}`。
- 未收齐：`409`，`{"detail":"上传未完成","received": n}`。
- 哈希不符：`422`，`{"detail":"文件校验失败，请重新上传"}`，并删除该上传（App 据此清掉 `uploadId`、提示重新上传）。
- finish 本身可重复调用：已完成的上传再 finish 返回同一个 `201` + `job_id`（App 收到 201 后进程被杀、再打开时会重发）。

`upload_id` 与之后的 `job_id` 是同一个值，目录也是同一个：`data/jobs/<id>/`。上传中目录里是 `capture.zip.part` + `upload.json`（size、sha256、received、created_at、updated_at），没有 `status.json`，所以状态页和 `recover_after_restart` 自动忽略它；finish 时把 part 改名为 `capture.zip`、删 `upload.json`、写 `status.json`（queued）并入队——这一步和现有 `Upload.commit` 走同一个 `_commit`。

`received` 以 `upload.json` 为准，每次 PUT 结束（正常或断开）时用实际文件大小更新并原子写入；服务启动时也用文件大小校正一次（上次进程被杀时可能没来得及写 json）。

过期：`upload.json` 的 `updated_at` 超过 24 小时的上传目录，在服务启动时和每次 `POST /uploads` 时清理；正在接收（锁被占）的上传跳过。

### GET /jobs/{id}/scene.splat

不变。Starlette 1.7 的 `FileResponse` 已支持 `Range`/`If-Range`，返回 `206` + `Content-Range`，响应头带 `Accept-Ranges: bytes`、`ETag`、`Last-Modified`。App 续传只需带 `Range: bytes=<.part 大小>-`。

## 4. App 侧

### `Api` 接口

```kotlin
/** 一次拍摄的上传：可断点续传，返回 job id。[uploadId] 为上次中断时持久化的 id（null = 新上传）。 */
suspend fun upload(zip: File, uploadId: String?, listener: UploadListener): String

interface UploadListener {
    /** 服务端分配了上传 id；调用方立刻持久化，进程被杀后用它续传。 */
    fun onCreated(uploadId: String)
    /** 已传字节占比 0..1（按服务端确认的 received 算，不是本地已写出的）。 */
    fun onProgress(fraction: Float)
    /** 第 [attempt] 次连续失败后开始等待重连；恢复后下一次 onProgress 即表示重连成功。 */
    fun onRetry(attempt: Int)
}

/** 下载 scene.splat 到 [dest]，`.part` 续传；[dest] 只在完整时出现。 */
suspend fun downloadScene(id: String, dest: File, onRetry: (Int) -> Unit)
```

### 上传流程（`ApiClient.upload`）

1. 算 zip 的 SHA-256（流式，100–200 MB 约 1 s）。
2. `uploadId == null` → `POST /uploads` 拿 id，`onCreated`；否则 `GET /uploads/{id}` 拿 `received`。404 时先 `GET /jobs/{id}`：200 说明上次 finish 已成功、进程在清 `uploadId` 前被杀，直接把该 id 当 job id 返回；也是 404 才当作新上传，从 1 重来并再次 `onCreated`。
3. 循环：从 `received` 读 4 MB 发 `PUT`，成功则 `received` = 响应值，`onProgress`。
4. 任何 `IOException` / 5xx：`attempt++`，`onRetry(attempt)`，退避等待（2 s × 2^(attempt−1)，上限 30 s），然后回到第 2 步的 `GET` 重新取 `received`。`409 偏移不匹配` 同样用响应里的 `received` 纠正后直接继续，不计失败。连续 30 次失败（约 15 分钟全程不可达）→ 抛 `ApiException(MSG_UNREACHABLE)`；一次成功的 PUT 把计数归零。
5. 收齐后 `POST /uploads/{id}/finish`；`422 校验失败` → 抛带该 detail 的 `ApiException`，并由 `UploadViewModel` 清掉 `uploadId`。

错误分类（`ApiException` 带 `code: Int?`，HTTP 状态码，传输错误为 null）：`IOException`、5xx、`409 正在接收中` → 可重试；`409 偏移不匹配` → 纠偏继续；`404 上传不存在` → 第 2 步的恢复逻辑；其他 4xx（413 超 size、422 校验失败、400）→ 致命，`UploadViewModel` 清掉 `uploadId`、标失败、按钮"重新上传"。

每个 PUT 用 `RequestBody` 直接从文件偏移读，不把 4 MB 读进内存（`FileInputStream.skip` + 定长 `Source`）。OkHttp 超时：connect 5 s、read/write 60 s 不变。

### 下载流程（`ApiClient.downloadScene`）

`.part` 存在则带 `Range: bytes=<len>-`：`206` 追加、`200` 从头覆盖、`416` 删 `.part` 重来。`IOException` 同上重试（同样 30 次上限）。完整后 rename 为 `dest`。不带 `If-Range`：同一 job id 的 `scene.splat` 导出后不再改变。下载中重连时记录 message = `正在下载模型… · 网络中断，正在重连 (n)`，行文案原样显示。

### `UploadViewModel`

- `CaptureRecord` 新增 `uploadId: String? = null`（`rooms.json` 已 `ignoreUnknownKeys`，老记录兼容）。
- `upload()`：把 `onCreated` 的 id 写进记录；`onRetry(n)` 时 message = `网络中断，正在重连 (n)`，行文案显示为 `上传中 37% · 网络中断，正在重连 (3)`；恢复后清 message。
- 成功拿到 `jobId` 后清 `uploadId`。
- `resumeInterrupted()`：`uploading` 且 `uploadId != null` → 直接 `start`（会续传）；`uploadId == null`（老版本的半截上传）→ 维持现有行为标失败。
- `retry()`：`jobId == null` 时 `start`，`start` 看到记录里有 `uploadId` 就续传——所以"重新上传"按钮在有 `uploadId` 时文案改为"继续上传"。
- 校验失败（finish 422）：清 `uploadId`，标失败，按钮回到"重新上传"。

## 5. 不变的东西

- 训练中断网的轮询重试（10 次、退避）不动。
- 拍摄包格式、任务状态机、状态页不动。
- 旧 `POST /jobs` 保留，`docs/api.md` 标为"旧接口（命令行/测试用）"。

## 6. 验收

- 单元：服务端 `tests/test_api.py` 覆盖创建 / 分块 / 偏移不匹配 / 断开后 `received` 保留 / 超 size 413 / 哈希不符 422 / finish 幂等 / 过期清理 / 启动校正 received；App `ApiClientTest`（MockWebServer）覆盖续传偏移、409 纠偏、断线重试、下载 206 追加；`UploadViewModelTest` 覆盖 uploadId 持久化与续传、重连文案、校验失败清 id。
- 真机（`docs/acceptance/phase1.md` §4 第 6 项补充）：家里 Wi-Fi 经 Tailscale 地址传一个 150 MB 包，中途开关飞行模式两次，看到"正在重连 (n)"后自动恢复、最终"完成"，服务端 `log.txt` 只有一条 created；训练中杀 App 再开仍能下载；上传中杀 App 再开从断点继续（logcat 里 `GET /uploads/{id}` 后紧接 `PUT` 的 `Upload-Offset` 非 0）。

## 7. 我的裁定（builder 可以不回来问）

- 上传 id 复用为 job id、同一目录：少一次跨目录移动（`/mnt/d` 上 9p 文件系统的 `rmtree` 曾静默失败，见 phase1 §5 问题 4），也让状态页零改动。
- 块大小 4 MB 固定、不自适应：服务端边收边写，块大小只影响请求数。
- 重试上限 30 次、退避 2 s→30 s：与现有轮询的 10 次/3 s→30 s 同一风格但更宽容，因为上传等的时间本来就长。
- SHA-256 必填：手机算一次 1 s，换来"错一个字节就拒绝"。
- 不做并发 PUT：链路本来就是瓶颈，并发只会让每个连接更容易断。
- 不比对本地 zip 是否变过：拍摄 id 带时间戳，zip 路径不复用。
