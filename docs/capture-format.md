# 拍摄包格式 v1

拍摄包是一个 zip 文件，下列文件位于 zip 根目录。

## manifest.json

```json
{"format_version":1,"capture_id":"cap-001","room_id":"room-001","kind":"model",
 "device":"Pixel 8","created_at":"2026-11-01T08:00:00Z",
 "image_width":640,"image_height":480,"depth_width":80,"depth_height":60,
 "frame_count":120}
```

- `format_version`: int，当前必须为 1，否则报“不支持的拍摄包版本”。
- `capture_id`, `room_id`, `device`: str；`kind` 固定为 `"model"`。
- `created_at`: ISO8601。
- `image_width/height`: JPEG 分辨率。
- `depth_width/height`: int 或 null（无深度时为 null）。
- `frame_count`: 必须等于 frames.jsonl 行数，否则报“帧数不一致”。

## frames.jsonl

每行一帧：

```json
{"index":1,"t_ns":33000000,"image":"images/000001.jpg","depth":"depth/000001.png",
 "pose":[1,0,0,0, 0,1,0,0, 0,0,1,0, 0.1,0,0,1],
 "fx":500.0,"fy":500.0,"cx":320.0,"cy":240.0,
 "depth_fx":62.5,"depth_fy":62.5,"depth_cx":40.0,"depth_cy":30.0}
```

- `depth`: 路径或 null。
- `pose`: 16 个 float，列主序，为 ARCore `Camera.getPose()` 的 world_from_camera，OpenGL 约定（相机朝 -Z，Y 向上）。读取方式：`np.array(pose).reshape(4,4).T`。
- `fx, fy, cx, cy`: 内参，对应 JPEG 分辨率。
- `depth_fx, depth_fy, depth_cx, depth_cy`: **可选** float，深度图自己的内参，单位为深度图像素（depth_width x depth_height）。
  只在该帧有深度时出现（新增于 v1 内，`format_version` 仍为 1，旧包没有这几项也合法）。
  原因：ARCore 深度图与 GPU 纹理对齐，而不一定与 CPU 图像对齐；App 取 `camera.getTextureIntrinsics()` 的焦距和主点，
  按 `textureIntrinsics.imageDimensions → depth_width/height` 的比例缩放后写入。
  缺失时，按图像内参缩放（见下面 depth 一节）。

## images/*.jpg

ARCore CPU 图像，传感器原始方向（横屏），不旋转。文件缺失报“缺少图片”。

## depth/*.png

16 位单通道，单位毫米，0 表示无效。分辨率为 depth_width x depth_height。

深度内参：帧里有 `depth_fx/fy/cx/cy` 时直接使用；没有时，假定与图像同视场，按比例缩放（fx * depth_width / image_width 等）。
如果没有深度内参，且深度图宽高比与图像宽高比相差超过 1%（|(dw/dh)/(W/H) − 1| > 0.01），服务端认为无法对齐，
跳过该帧的深度（记一行日志，不报错），该帧只用于颜色训练。

## 参考实现

`server/recon3d/capture/format.py` 中的 `load_capture(zip_path, workdir) -> Capture`；格式错误抛 `CaptureFormatError(message_zh)`。测试用生成器：`server/tests/fixtures/make_capture.py`。
