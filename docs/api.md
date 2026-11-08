# 服务端 API（前缀 `/api/v1`）

**状态页**：浏览器打开 `http://<服务端地址>/`（不带 `/api/v1`，例如笔记本上 `http://127.0.0.1:8765/`），
列出所有任务的状态、模式（快速 / 高精）、进度、说明、创建时间、拍摄包大小，完成的任务可直接下载 `scene.splat`，每 5 秒自动刷新。页面没有删除按钮；电脑上删任务：
`curl -X DELETE http://localhost:8765/api/v1/jobs/<id>`。

**重建模式 `mode`**：`"fast"`（默认，直接用 ARCore 位姿训练）或 `"precise"`（先用 COLMAP 优化位姿再训练，
见 `docs/superpowers/specs/2026-11-05-recon3d-precise-mode-design.md`）。上传时指定（`POST /uploads` / `POST /jobs`），
或对已有任务用 `POST /jobs/{id}/rerun` 换模式重算。所有返回任务的接口都带 `mode` 字段；`mode` 不是这两个值 → `422`。

## GET /health

```json
{"version":"0.1.0","gpu":"NVIDIA GeForce RTX 4060","profile":"laptop"}
```

`profile`: `"laptop"` 或 `"server"`。

## POST /jobs（旧接口：命令行 / 测试用，手机不再使用）

multipart 表单，字段 `capture` 为拍摄包 zip（见 capture-format.md）；可选文本字段 `mode`（`fast` | `precise`，默认 `fast`）。

响应 `201`：`{"job_id":"3f9c0a1b2d4e"}`（12 位十六进制）

错误：
- `413`：拍摄包超过服务端上限（默认 4 GB，`--max-upload-gb` 可调），`{"detail":"上传文件过大，上限 4 GB"}`
- `422`：缺少 `capture` 字段，或请求不是 `multipart/form-data`；`mode` 非法：`{"detail":"mode 只能是 fast 或 precise"}`
- `400`：上传不完整或 multipart 格式错误

上传按流式解析直接写入磁盘；无 `Content-Length`（分块传输）时同样在超过上限的瞬间返回 413。

## GET /jobs

所有任务，最新的在前（给状态页用，手机不用）：

```json
{"jobs":[{"job_id":"3f9c0a1b2d4e","state":"done","progress":1.0,"message":"完成",
          "created_at":1793628538.96,"capture_bytes":71075878,"has_scene":true,"mode":"fast"}]}
```

正在接收的断点续传上传排在最前面，`state` 为 `uploading`，`progress` = 已收/总大小，
`message` 为 `已收到 n / size 字节`，`capture_bytes` 是声明的总大小，`mode` 是 `POST /uploads` 时声明的模式。finish 后变成普通任务。

## GET /jobs/{id}

```json
{"job_id":"3f9c0a1b2d4e","state":"training","progress":0.42,"message":"训练中","mode":"fast",
 "objects":{"state":"none","progress":0.0,"message":"","count":0}}
```

- `state`: `queued` | `training` | `exporting` | `done` | `failed`
- `progress`: 0..1。快速：训练 0→0.9，导出 0.9→1.0。高精：位姿优化 0→0.3，训练 0.3→0.9，导出 0.9→1.0
- `message`: 中文状态说明；失败时为错误原因（如"显存不足，请降低分辨率或改用服务器"、"服务重启，任务中断，请重新上传"）。
  高精模式位姿阶段 `state` 仍为 `training`，`message` 依次为 `位姿优化 (特征提取)` / `位姿优化 (特征匹配)` / `位姿优化 (全局优化)`；
  位姿优化失败时自动按快速模式重建，完成后 `message` 为 `完成（高精位姿失败，已按快速模式重建）`，原因在服务端 `log.txt`。
- `mode`: `fast` | `precise`（老任务的 status.json 没有该字段时按 `fast`）。
- `objects`: 物体识别（第二期）状态块，`GET /jobs` 的每行里也有。`state`: `none`（没跑过）| `queued` | `running` | `done` | `failed`；
  `progress` 0..1（检测 0→0.6，合并 →0.75，归属 →0.95，截图 →1.0）；`message` 运行中为 `识别物体 (检测)` / `(合并)` / `(归属)` / `(截图)`，
  失败时为原因（`显存不足，请降低分辨率或改用服务器`、`模型下载失败，请检查网络`、`识别失败，请重试；…`）；`count` 完成后的物体数。
  服务重启时 `queued` / `running` 的识别回到 `queued` 自动续跑；任务本身的 `state` 不受识别影响。

任务不存在：`404`，`{"detail":"任务不存在"}`。

## POST /jobs/{id}/rerun

对已有任务的拍摄包换模式重算（手机端"改用高精重算 / 改用快速重算"）。请求 JSON：`{"mode":"precise"}`。
新任务目录里的 `capture.zip` 是源任务文件的硬链接（不占额外磁盘；文件系统不支持时复制），源任务不变、可同时在训练。

- 成功：`201`，`{"job_id":"<新任务 id>"}`，新任务 `queued`
- 源任务不存在 / 没有拍摄包：`404`，`{"detail":"任务不存在"}`
- id 是还没 finish 的断点续传上传：`409`，`{"detail":"上传未完成"}`
- `mode` 非法：`422`

## POST /jobs/{id}/objects

对已完成的任务按需跑物体识别（手机"识别物体"按钮）。无请求体。结果已存在时重新识别并覆盖。

- 成功：`202`，`{"job_id":"…","objects":{"state":"queued",…}}`
- 任务不存在：`404`，`{"detail":"任务不存在"}`
- 任务还没 `done` 或没有 `capture.zip`：`409`，`{"detail":"任务尚未完成"}`
- 已在排队 / 识别中：`409`，`{"detail":"正在识别"}`

识别中的任务不能删除（`DELETE` 返回 `409`，同训练中）。

## GET /jobs/{id}/objects.zip

物体表产物包（`application/zip`），`objects.state == done` 时可下载，否则 `404`，`{"detail":"没有物体表"}`。内容：

- `objects.json`：
  ```json
  {"version": 2, "job_id": "…", "frames_used": 1123, "gaussians": 983112,
   "objects": [{"id": 1, "label": "chair", "label_zh": "椅子", "score": 0.64, "views": 17,
                "center": [x, y, z], "bbox": {"min": [x, y, z], "max": [x, y, z]},
                "gaussian_count": 23410, "thumb": "thumbs/1.jpg",
                "level": "furniture", "size": [0.6, 0.6, 1.0], "support": "floor", "confidence": 0.75}],
   "relations": [{"a": 3, "rel": "on", "b": 7}],
   "structure": {"floor_y": 0.0, "ceiling_y": 2.6,
                 "walls": [{"id": 1, "normal": [nx, 0, nz], "d": 2.0, "u": [ux, 0, uz], "extent": [u0, u1, y0, y1]}]}}
  ```
  `id` 从 1 起连续；坐标为场景世界系（与 `scene.splat` 一致，+Y 向上）；`rel` ∈ `on` / `in` / `next_to` / `left_of` / `right_of`。
  v2 新增：`level` ∈ `structure`（地面、天花板、墙面、门、窗）/ `furniture` / `item`；`size` 为重力对齐的长、宽、高（米）；
  `support` ∈ `floor` / `ceiling` / `on:<id>` / `wall:<id>` / `none`；`confidence` 0–1（提议器与最终标签一致的帧占比）；
  `views` 为该物体被跟踪到的帧数。`structure` 为房间结构：墙的平面方程 `normal · p + d = 0`（法向朝室内），
  `u` 为墙面内的水平轴，`extent` 为 `[u_min, u_max, y_min, y_max]`。版本 1 的表（无这些字段）手机端仍可打开。
- `objects.bin`：`gaussians` 个 little-endian `uint16`，第 i 个是 `scene.splat` 第 i 个高斯所属物体 `id`，0 = 无。
- `thumbs/{id}.jpg`：代表截图（长边 256）。

## GET /jobs/{id}/scene.splat

`Content-Type: application/octet-stream`。仅当 `state == "done"` 时可用。支持 `Range`（响应 `206` + `Content-Range`，
头带 `Accept-Ranges: bytes`），手机下载中断后带 `Range: bytes=<已有字节数>-` 续传。

- 任务存在但未完成（含失败）：`409`，`{"detail":"任务尚未完成"}`
- 任务不存在：`404`，`{"detail":"任务不存在"}`

## DELETE /jobs/{id}

删除任务目录（拍摄包、结果、日志）；对正在接收的断点续传上传同样有效（删掉半截文件）。手机端"彻底删除"用它。

- 成功：`204`，无响应体
- 训练中 / 导出中（GPU 正往目录里写）：`409`，`{"detail":"任务正在训练，完成后再删"}`
- 正有 PUT 在写这个上传：`409`，`{"detail":"该上传正在接收中","received":<已收字节数>}`（手机取消上传后隔 1 s 重试）
- 不存在或 id 格式非法：`404`，`{"detail":"任务不存在"}`

排队中的任务可删；worker 取到已删的 id 直接跳过。

## 断点续传上传（手机用）

拍摄包按块上传，断线后从服务端已收到的字节处继续；`upload_id` 与之后的 `job_id` 是同一个值。

### POST /uploads

请求 JSON：`{"size": 123456789, "sha256": "<64 位十六进制>", "mode": "fast"}`（`mode` 可省，默认 `fast`）。
响应 `201`：`{"upload_id":"3f9c0a1b2d4e","received":0}`。

- `413`：`size` 超过上限，`{"detail":"上传文件过大，上限 4 GB"}`
- `422`：`size` ≤ 0、`sha256` 格式不对，或 `mode` 不是 `fast` / `precise`

每次调用先清理 24 小时无活动的未完成上传。

### PUT /uploads/{id}

请求头 `Upload-Offset: <已收字节数>`，请求体 `application/octet-stream`，任意长度（手机用 4 MB）。
服务端边收边写：连接中途断开，已写入的字节保留；连续 30 s 收不到请求体视为断开。
响应 `200`：`{"received": <写完后的总字节数>}`。

- `409` `{"detail":"偏移不匹配","received": n}`：`Upload-Offset` 与已收字节数不符，按 `received` 纠正后重发
- `409` `{"detail":"该上传正在接收中","received": n}`：同一上传另一个 PUT 还在处理，稍后重试
- `409` `{"detail":"上传已完成"}`：已经 finish 过
- `413`：写入后超过声明的 `size`，该上传被删除
- `422` `{"detail":"缺少 Upload-Offset"}`
- `404` `{"detail":"上传不存在"}`

### GET /uploads/{id}

响应 `200`：`{"upload_id":"3f9c0a1b2d4e","size":123456789,"received":4194304}`。
不存在或已 finish：`404`（已 finish 时 `GET /jobs/{id}` 会返回任务）。

### POST /uploads/{id}/finish

收齐且 SHA-256 一致 → 创建任务并入队，响应 `201`：`{"job_id":"3f9c0a1b2d4e"}`。可重复调用，已完成的上传再 finish 返回同一个 `job_id`。

- `409` `{"detail":"上传未完成","received": n}`
- `422` `{"detail":"文件校验失败，请重新上传"}`：该上传被删除
- `404` `{"detail":"上传不存在"}`

以上错误响应体均为 `{"detail": "<中文说明>"}`（续传接口的 `409` 另带 `received`）。
