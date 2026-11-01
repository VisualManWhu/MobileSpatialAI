# 服务端 API（前缀 `/api/v1`）

## GET /health

```json
{"version":"0.1.0","gpu":"NVIDIA GeForce RTX 4060","profile":"laptop"}
```

`profile`: `"laptop"` 或 `"server"`。

## POST /jobs

multipart 表单，字段 `capture` 为拍摄包 zip（见 capture-format.md）。

响应 `201`：`{"job_id":"j-abc123"}`

## GET /jobs/{id}

```json
{"job_id":"j-abc123","state":"training","progress":0.42,"message":"训练中"}
```

- `state`: `queued` | `training` | `exporting` | `done` | `failed`
- `progress`: 0..1
- `message`: 中文状态说明；失败时为错误原因。

## GET /jobs/{id}/scene.splat

`Content-Type: application/octet-stream`。仅当 `state == "done"` 时可用，否则返回错误。
