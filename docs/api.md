# 服务端 API（前缀 `/api/v1`）

**状态页**：浏览器打开 `http://<服务端地址>/`（不带 `/api/v1`，例如笔记本上 `http://127.0.0.1:8765/`），
列出所有任务的状态、进度、说明、创建时间、拍摄包大小，完成的任务可直接下载 `scene.splat`，每 5 秒自动刷新。

## GET /health

```json
{"version":"0.1.0","gpu":"NVIDIA GeForce RTX 4060","profile":"laptop"}
```

`profile`: `"laptop"` 或 `"server"`。

## POST /jobs

multipart 表单，字段 `capture` 为拍摄包 zip（见 capture-format.md）。

响应 `201`：`{"job_id":"3f9c0a1b2d4e"}`（12 位十六进制）

错误：
- `413`：拍摄包超过服务端上限（默认 4 GB，`--max-upload-gb` 可调），`{"detail":"上传文件过大，上限 4 GB"}`
- `422`：缺少 `capture` 字段，或请求不是 `multipart/form-data`
- `400`：上传不完整或 multipart 格式错误

上传按流式解析直接写入磁盘；无 `Content-Length`（分块传输）时同样在超过上限的瞬间返回 413。

## GET /jobs

所有任务，最新的在前（给状态页用，手机不用）：

```json
{"jobs":[{"job_id":"3f9c0a1b2d4e","state":"done","progress":1.0,"message":"完成",
          "created_at":1793628538.96,"capture_bytes":71075878,"has_scene":true}]}
```

## GET /jobs/{id}

```json
{"job_id":"3f9c0a1b2d4e","state":"training","progress":0.42,"message":"训练中"}
```

- `state`: `queued` | `training` | `exporting` | `done` | `failed`
- `progress`: 0..1（训练 0→0.9，导出 0.9→1.0）
- `message`: 中文状态说明；失败时为错误原因（如"显存不足，请降低分辨率或改用服务器"、"服务重启，任务中断，请重新上传"）。

任务不存在：`404`，`{"detail":"任务不存在"}`。

## GET /jobs/{id}/scene.splat

`Content-Type: application/octet-stream`。仅当 `state == "done"` 时可用。

- 任务存在但未完成（含失败）：`409`，`{"detail":"任务尚未完成"}`
- 任务不存在：`404`，`{"detail":"任务不存在"}`

以上错误响应体均为 `{"detail": "<中文说明>"}`。
