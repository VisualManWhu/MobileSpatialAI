# Recon3D 第一期 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Pixel 10 Pro 上按箭头引导（风格 A）拍一个房间 → 局域网上传到笔记本 → gsplat 训练 3DGS → 回传并在手机上浏览。

**Architecture:** 仓库分 `server/`（Python，运行在 Ubuntu：笔记本上的 WSL2 Ubuntu 和实验室 A6000 服务器是同一套环境）和 `android/`（Kotlin + Compose + ARCore）。两边只通过一个带版本号的"拍摄包"zip 和一组 HTTP 接口耦合，二者在 Task 1 先定死。手机端 3D 浏览用 WebView 加载打包进 App 的 Spark（three.js 系 3DGS 查看器），场景格式为 `.splat`。

**Tech Stack:** Python 3.11、uv、PyTorch（CUDA 12.x）、gsplat 1.5.x、FastAPI、uvicorn、pytest；Kotlin 2.x、Jetpack Compose、ARCore（`com.google.ar:core`）、OkHttp、kotlinx.serialization、JUnit4、MockWebServer；Spark（`@sparkjsdev/spark`）+ three.js。

**Spec:** `docs/superpowers/specs/2026-11-01-recon3d-design.md`

## Global Constraints

- 手机：Google Pixel 10 Pro；`minSdk 33`，`targetSdk`/`compileSdk` 用安装时最新稳定版；需 ARCore + Depth API。
- 笔记本：Windows 11，RTX 4060 Laptop **8 GB** 显存，32 GB 内存；服务端跑在已有的 **WSL2 Ubuntu** 中（本机已装，无 Docker、无 Java、无 Android SDK）。
- 服务器：RTX A6000 Ada **48 GB**，Ubuntu，在实验室。第一期只要求同一套安装脚本能在上面跑；笔记本 → 服务器转发留到第二期。
- 服务端代码在两台机器上只靠配置档切换：`profile=laptop` / `profile=server`。
- 网络：手机与计算端同一 Wi-Fi；所有依赖可直接从官方源下载（用户在德国）。
- 一条命令启动服务端；用户不手动编译 CUDA 扩展（gsplat 用官方预编译 wheel）。
- 所有面向用户的文字（App 界面、错误提示、安装文档）用**中文**。
- 拍摄包格式带 `format_version`，本期为 `1`。
- 版本钉死：每个依赖在首次引入时取当时最新稳定版，写入 `server/uv.lock` / `android/gradle/libs.versions.toml`，之后不再浮动。

## Review Focus

1. **拍摄中 ARCore 跟丢 / 用户在阶段①未完成时想停止** → 跟丢期间的帧绝不写入拍摄包；阶段①未完成时"完成拍摄"按钮不可用，并提示原因。（测试在 Task 9、Task 10）
2. **上传时手机断网或服务端没开** → 显示中文错误，拍摄包保留在手机上，可点"重新上传"。（测试在 Task 12）
3. **服务端重启时有任务在排队或训练** → 重启后这些任务标为 `failed`，消息为"服务重启，任务中断，请重新上传"，不会永远卡在"训练中"。（测试在 Task 6）
4. **白墙多、深度稀疏，初始化点云几乎为空** → 不崩溃，退回到在相机视锥范围内随机初始化。（测试在 Task 3）
5. **笔记本显存不够（CUDA OOM）** → 任务 `failed`，消息为"显存不足，请降低分辨率或改用服务器"，worker 继续处理下一个任务。（测试在 Task 6）

---

## 文件结构

```
.gitignore
README.md                         中文：项目是什么、文档入口
docs/
  capture-format.md               拍摄包 v1 格式（Task 1 定死）
  api.md                          HTTP 接口（Task 1 定死）
  setup/01-server-laptop.md       笔记本（WSL2）安装
  setup/02-server-lab.md          实验室服务器安装
  setup/03-phone.md               手机开发者模式 + 安装 App
  acceptance/phase1.md            第一期人工验收清单
server/
  pyproject.toml, uv.lock
  scripts/install.sh              Ubuntu 一键安装（WSL2 与实验室共用）
  scripts/run.sh                  一键启动：uv run recon3d-server --profile ...
  recon3d/
    config.py                     Profile 数据类 + laptop/server 两档
    capture/format.py             拍摄包解析与校验
    capture/poses.py              ARCore(OpenGL) → OpenCV 位姿转换
    recon/dataset.py              帧 → 训练张量（缩放、内参）
    recon/init_points.py          深度反投影初始化点云
    recon/trainer.py              gsplat 训练循环
    recon/export_splat.py         导出 .splat
    jobs/store.py                 任务目录 + status.json
    jobs/worker.py                单 GPU 串行 worker
    api/app.py                    FastAPI
    main.py                       CLI 入口
  tests/ ...                      与模块一一对应；GPU 测试标 @pytest.mark.gpu
android/
  settings.gradle.kts, build.gradle.kts, gradle/libs.versions.toml
  app/src/main/java/dev/recon3d/
    MainActivity.kt, nav/AppNav.kt
    settings/ServerSettings.kt    计算端地址存储（DataStore）
    net/ApiClient.kt              健康检查 / 上传 / 查询 / 下载
    rooms/RoomRepository.kt       本地房间列表（JSON 文件）
    rooms/RoomListScreen.kt
    capture/PoseMath.kt           yaw/pitch、角速度
    capture/CaptureGuide.kt       三阶段引导状态机（纯 Kotlin）
    capture/KeyframeSelector.kt
    capture/CapturePackageWriter.kt
    capture/ar/ArSessionHelper.kt 相机配置选择、session 生命周期
    capture/ar/BackgroundRenderer.kt  相机画面渲染（改自 ARCore 官方示例，Apache-2.0）
    capture/CaptureScreen.kt      AR 画面 + 风格 A 覆盖层
    upload/UploadScreen.kt        上传与进度
    viewer/ViewerScreen.kt        WebView + WebViewAssetLoader
  app/src/main/assets/viewer/     index.html + 打包好的 spark/three
  app/src/test/java/dev/recon3d/  单元测试
```

---

### Task 1: 仓库骨架 + 两份接口契约

**Files:**
- Create: `README.md`, `docs/capture-format.md`, `docs/api.md`, `server/pyproject.toml`, `server/recon3d/__init__.py`, `server/recon3d/capture/format.py`
- Test: `server/tests/test_capture_format.py`, `server/tests/fixtures/make_capture.py`

**Interfaces:**
- Produces（拍摄包 v1，zip，根目录）：
  - `manifest.json`：`{"format_version":1,"capture_id":str,"room_id":str,"kind":"model","device":str,"created_at":ISO8601,"image_width":int,"image_height":int,"depth_width":int|null,"depth_height":int|null,"frame_count":int}`
  - `frames.jsonl`：每行 `{"index":int,"t_ns":int,"image":"images/000001.jpg","depth":"depth/000001.png"|null,"pose":[16 floats, 列主序, ARCore Camera.getPose() 的 world_from_camera, OpenGL 约定],"fx":f,"fy":f,"cx":f,"cy":f}`，内参对应 JPEG 分辨率。
  - `images/*.jpg`：ARCore CPU 图像，传感器原始方向（横屏），不旋转。
  - `depth/*.png`：16 位单通道，单位毫米，0 表示无效；分辨率为 depth_width×depth_height，与图像同视场，内参按比例缩放。
  - API（`docs/api.md`，前缀 `/api/v1`）：`GET /health` → `{"version":str,"gpu":str,"profile":"laptop"|"server"}`；`POST /jobs`（multipart 字段 `capture`）→ `201 {"job_id":str}`；`GET /jobs/{id}` → `{"job_id","state":"queued"|"training"|"exporting"|"done"|"failed","progress":0..1,"message":str}`；`GET /jobs/{id}/scene.splat` → `application/octet-stream`（仅 done）。
  - Python：`load_capture(zip_path: Path, workdir: Path) -> Capture`；`Capture(manifest: Manifest, frames: list[Frame], root: Path)`；格式错误抛 `CaptureFormatError(message_zh: str)`。
  - 测试工具：`make_capture(tmp_path, n_frames=4, with_depth=True, **overrides) -> Path`

- [ ] **Step 1:** `git init` 已有；写 `docs/capture-format.md`、`docs/api.md`，内容即上面 Interfaces 原文加一个示例。
- [ ] **Step 2:** `cd server && uv init --package` 生成 `pyproject.toml`（Python 3.11），加依赖 `pydantic`、`numpy`、`pillow`，dev 依赖 `pytest`。
- [ ] **Step 3: 写失败测试**

```python
def test_load_valid_capture(tmp_path):
    cap = load_capture(make_capture(tmp_path, n_frames=4), tmp_path / "w")
    assert cap.manifest.format_version == 1
    assert len(cap.frames) == 4
    assert cap.frames[0].pose.shape == (4, 4)

def test_rejects_unknown_version(tmp_path):
    with pytest.raises(CaptureFormatError, match="不支持的拍摄包版本"):
        load_capture(make_capture(tmp_path, format_version=2), tmp_path / "w")

def test_rejects_missing_image(tmp_path):
    with pytest.raises(CaptureFormatError, match="缺少图片"):
        load_capture(make_capture(tmp_path, drop_image=1), tmp_path / "w")

def test_frame_count_mismatch(tmp_path):
    with pytest.raises(CaptureFormatError, match="帧数不一致"):
        load_capture(make_capture(tmp_path, frame_count=9), tmp_path / "w")
```

- [ ] **Step 4:** `uv run pytest tests/test_capture_format.py -v` → FAIL（ImportError）。
- [ ] **Step 5:** 实现 `format.py`（pydantic 模型；pose 读为列主序 → `np.array(...).reshape(4,4).T`）和 `make_capture`。
- [ ] **Step 6:** 再跑 → 4 passed。
- [ ] **Step 7:** Commit `feat(server): capture format v1 and API contract`。

---

### Task 2: 位姿转换 + 训练数据集

**Files:**
- Create: `server/recon3d/capture/poses.py`, `server/recon3d/recon/dataset.py`, `server/recon3d/config.py`
- Test: `server/tests/test_poses.py`, `server/tests/test_dataset.py`

**Interfaces:**
- Consumes: `Capture`、`Frame`（Task 1）
- Produces:
  - `gl_to_cv_c2w(pose_gl: np.ndarray) -> np.ndarray`：`pose_gl @ diag(1,-1,-1,1)`
  - `Profile(name, max_image_side:int, iterations:int, max_gaussians:int, sh_degree:int)`；`PROFILES = {"laptop": Profile("laptop", 960, 15000, 1_500_000, 1), "server": Profile("server", 1600, 30000, 4_000_000, 3)}`
  - `TrainingSet(images: Tensor[N,H,W,3] float 0..1, K: Tensor[N,3,3], c2w: Tensor[N,4,4], depths: list[Tensor|None])`；`build_training_set(cap: Capture, profile: Profile, device="cpu") -> TrainingSet`

- [ ] **Step 1: 写失败测试**

```python
def test_gl_identity_looks_down_minus_z():
    c2w = gl_to_cv_c2w(np.eye(4))
    assert np.allclose(c2w[:3, 2], [0, 0, -1])   # CV +Z (forward) == GL -Z
    assert np.allclose(c2w[:3, 1], [0, -1, 0])   # CV +Y (down)

def test_downscale_scales_intrinsics(tmp_path):
    cap = load_capture(make_capture(tmp_path, image_width=1920, image_height=1080, fx=1500), tmp_path/"w")
    ts = build_training_set(cap, PROFILES["laptop"])
    assert ts.images.shape[2] == 960
    assert ts.K[0, 0, 0] == pytest.approx(750)
```

- [ ] **Step 2:** 跑 → FAIL。
- [ ] **Step 3:** 实现。缩放用 Pillow `LANCZOS`，内参 fx,fy,cx,cy 同比例缩放。
- [ ] **Step 4:** 跑 → PASS。
- [ ] **Step 5:** Commit `feat(server): pose conversion and training set`。

---

### Task 3: 深度反投影初始化点云

**Files:**
- Create: `server/recon3d/recon/init_points.py`
- Test: `server/tests/test_init_points.py`

**Interfaces:**
- Consumes: `TrainingSet`（Task 2）
- Produces: `init_points(ts: TrainingSet, voxel_size: float = 0.02, min_points: int = 5000, seed: int = 0) -> tuple[np.ndarray[M,3] xyz, np.ndarray[M,3] rgb 0..1]`
  - 每帧深度（毫米 → 米，0 丢弃，>8 m 丢弃）按缩放后的深度内参反投影，用 CV c2w 变到世界坐标，颜色取对应 RGB 像素；体素下采样（每体素取均值）。
  - 有效点数 `< min_points` 时：在所有相机位置包围盒向外扩 1 m 的范围内均匀随机补到 `min_points` 个点，颜色 0.5。

- [ ] **Step 1: 写失败测试**

```python
def test_plane_depth_backprojects_to_plane(ts_plane_2m):   # fixture: 相机在原点朝 CV +Z，深度全 2000mm
    xyz, rgb = init_points(ts_plane_2m, voxel_size=0.05)
    assert np.allclose(xyz[:, 2], 2.0, atol=0.03)

def test_empty_depth_falls_back_to_random(ts_no_depth):
    xyz, rgb = init_points(ts_no_depth, min_points=5000)
    assert xyz.shape == (5000, 3)
    assert np.all(np.isfinite(xyz))
```

- [ ] **Step 2:** 跑 → FAIL。
- [ ] **Step 3:** 实现（纯 numpy；体素下采样用 `np.unique(floor(xyz/voxel), axis=0, return_inverse=True)` 后 `np.add.at` 求均值）。
- [ ] **Step 4:** 跑 → PASS。
- [ ] **Step 5:** Commit `feat(server): depth-based point initialization`。

---

### Task 4: gsplat 训练器

**Files:**
- Create: `server/recon3d/recon/trainer.py`
- Modify: `server/pyproject.toml`（加 `torch`、`gsplat`，用 gsplat 官方预编译 wheel 索引，匹配 torch/CUDA 版本）
- Test: `server/tests/test_trainer.py`（`@pytest.mark.gpu`）

**Interfaces:**
- Consumes: `TrainingSet`、`Profile`、`init_points`
- Produces:
  - `Splats`：dataclass，字段 `means[N,3]`、`scales[N,3]`（log）、`quats[N,4]`（wxyz）、`opacities[N]`（logit）、`sh0[N,1,3]`、`shN[N,K,3]`，全部 torch Tensor。
  - `train(ts: TrainingSet, profile: Profile, on_progress: Callable[[float], None], device="cuda") -> Splats`
  - 损失 `0.8*L1 + 0.2*(1-SSIM)`；致密化用 `gsplat.DefaultStrategy`；高斯数达到 `profile.max_gaussians` 后停止致密化；`on_progress(i/iterations)` 每 100 步调用。
  - CUDA OOM 不在这里吞掉，原样抛出（Task 6 处理）。

- [ ] **Step 1: 写失败测试**——合成场景：用 gsplat `rasterization` 从 2000 个已知彩色高斯、绕圈 24 个相机渲染出图像，构造 `TrainingSet`；训练 2000 步后在 4 个留出视角上计算 PSNR。

```python
@pytest.mark.gpu
def test_trains_synthetic_scene(synthetic_ts):
    train_ts, test_ts = synthetic_ts
    splats = train(train_ts, dataclasses.replace(PROFILES["laptop"], iterations=2000), on_progress=lambda p: None)
    assert psnr(render(splats, test_ts), test_ts.images) > 25.0

@pytest.mark.gpu
def test_respects_max_gaussians(synthetic_ts):
    p = dataclasses.replace(PROFILES["laptop"], iterations=2000, max_gaussians=3000)
    assert train(synthetic_ts[0], p, on_progress=lambda p: None).means.shape[0] <= 3000
```

- [ ] **Step 2:** 在 WSL2 中 `uv run pytest -m gpu tests/test_trainer.py -v` → FAIL。
- [ ] **Step 3:** 实现训练循环（参照 gsplat `examples/simple_trainer.py` 的参数初始化与学习率；`render()` 辅助函数也放在 `trainer.py` 供测试和后续使用）。
- [ ] **Step 4:** 再跑 → 2 passed；记录 4060 上 2000 步耗时到提交说明。
- [ ] **Step 5:** Commit `feat(server): gsplat trainer with profiles`。

---

### Task 5: 导出 .splat

**Files:**
- Create: `server/recon3d/recon/export_splat.py`
- Test: `server/tests/test_export_splat.py`

**Interfaces:**
- Consumes: `Splats`（Task 4）
- Produces: `export_splat(splats: Splats, path: Path, min_opacity: float = 0.005) -> int`（返回写出的高斯数）
  - 每个高斯 32 字节、小端：`position f32×3`、`scale f32×3`（`exp(scales)`）、`rgba u8×4`（`rgb = clip(0.5 + 0.28209479*sh0, 0, 1)*255`，`a = sigmoid(opacity)*255`）、`rot u8×4`（归一化 wxyz，`q*128+128` 截断到 0..255）。
  - 去掉 `sigmoid(opacity) < min_opacity` 的点；按 `exp(sum(scales))*sigmoid(opacity)` 降序排序。
  - 训练结束同时保存完整 `scene.ply`（标准 3DGS ply 字段），留给第二期使用。

- [ ] **Step 1: 写失败测试**

```python
def test_splat_record_layout(tmp_path):
    s = make_splats(n=3, opacity_logit=[5.0, 5.0, -10.0])
    n = export_splat(s, tmp_path / "a.splat")
    data = (tmp_path / "a.splat").read_bytes()
    assert n == 2 and len(data) == 2 * 32
    pos = np.frombuffer(data[:12], "<f4")
    assert np.allclose(pos, s.means[0].numpy())
    assert data[12 + 12 + 3] == round(torch.sigmoid(torch.tensor(5.0)).item() * 255)
```

- [ ] **Step 2:** 跑 → FAIL。
- [ ] **Step 3:** 实现（numpy structured array 一次写出）。
- [ ] **Step 4:** 跑 → PASS。
- [ ] **Step 5:** Commit `feat(server): export .splat for mobile viewer`。

---

### Task 6: 任务存储、worker、HTTP 接口

**Files:**
- Create: `server/recon3d/jobs/store.py`, `server/recon3d/jobs/worker.py`, `server/recon3d/api/app.py`, `server/recon3d/main.py`
- Modify: `server/pyproject.toml`（`fastapi`、`uvicorn`、`python-multipart`、dev `httpx`；`[project.scripts] recon3d-server = "recon3d.main:cli"`）
- Test: `server/tests/test_jobs.py`, `server/tests/test_api.py`

**Interfaces:**
- Consumes: `load_capture`、`build_training_set`、`init_points`、`train`、`export_splat`
- Produces:
  - 数据目录 `<data_dir>/jobs/<job_id>/{capture.zip, status.json, scene.splat, scene.ply, log.txt}`；`job_id` 为 `uuid4().hex[:12]`。
  - `JobStore(data_dir)`：`create(zip_bytes_stream) -> str`、`get(job_id) -> JobStatus | None`、`update(job_id, **fields)`、`recover_after_restart() -> None`（把 `queued`/`training`/`exporting` 改为 `failed`，消息"服务重启，任务中断，请重新上传"）。
  - `Worker(store, pipeline: Callable[[Path, Callable[[float],None]], None])`：后台线程串行处理；`pipeline` 抛异常时任务 `failed`：`torch.cuda.OutOfMemoryError` → "显存不足，请降低分辨率或改用服务器"；`CaptureFormatError` → 其中文消息；其他 → "重建失败：" + 异常类型名；worker 继续处理下一个。进度：训练 0→0.9，导出 0.9→1.0。
  - `create_app(store, worker, profile) -> FastAPI`，实现 `docs/api.md`。
  - CLI：`recon3d-server --profile laptop|server --data-dir PATH --host 0.0.0.0 --port 8765`。

- [ ] **Step 1: 写失败测试**（用假 pipeline，不需要 GPU）

```python
def test_jobs_run_in_order(store):
    order = []
    w = Worker(store, pipeline=lambda d, p: order.append(d.name)); w.start()
    a, b = store.create(io.BytesIO(b"a")), store.create(io.BytesIO(b"b"))
    wait_until(lambda: store.get(b).state == "done")
    assert order == [a, b]

def test_oom_marks_failed_and_continues(store):
    calls = iter([torch.cuda.OutOfMemoryError(), None])
    def pipe(d, p):
        e = next(calls)
        if e: raise e
    w = Worker(store, pipe); w.start()
    a, b = store.create(io.BytesIO(b"a")), store.create(io.BytesIO(b"b"))
    wait_until(lambda: store.get(b).state == "done")
    assert store.get(a).state == "failed" and "显存不足" in store.get(a).message

def test_restart_recovery(store):
    j = store.create(io.BytesIO(b"a")); store.update(j, state="training")
    JobStore(store.data_dir).recover_after_restart()
    assert store.get(j).state == "failed" and "服务重启" in store.get(j).message

def test_api_upload_status_download(client):          # TestClient，假 pipeline 写一个 64 字节 scene.splat
    r = client.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")})
    assert r.status_code == 201
    jid = r.json()["job_id"]
    wait_until(lambda: client.get(f"/api/v1/jobs/{jid}").json()["state"] == "done")
    assert len(client.get(f"/api/v1/jobs/{jid}/scene.splat").content) == 64

def test_download_before_done_is_409(client_slow):
    jid = client_slow.post("/api/v1/jobs", files={"capture": ("c.zip", b"zip")}).json()["job_id"]
    assert client_slow.get(f"/api/v1/jobs/{jid}/scene.splat").status_code == 409

def test_unknown_job_404(client):
    assert client.get("/api/v1/jobs/nope").status_code == 404
```

- [ ] **Step 2:** 跑 → FAIL。
- [ ] **Step 3:** 实现。上传用流式写盘（不整包读入内存，拍摄包可达数百 MB）；`status.json` 先写临时文件再 `os.replace`。
- [ ] **Step 4:** 跑 → 全部 PASS。
- [ ] **Step 5:** Commit `feat(server): job queue and HTTP API`。

---

### Task 7: 一键安装 / 启动 + 笔记本局域网可达

**Files:**
- Create: `server/scripts/install.sh`, `server/scripts/run.sh`, `docs/setup/01-server-laptop.md`, `docs/setup/02-server-lab.md`
- Test: `server/tests/test_e2e_gpu.py`（`@pytest.mark.gpu`）

**Interfaces:**
- Consumes: 全部 server 模块
- Produces: `bash server/scripts/install.sh`（安装 uv → `uv sync` → 运行 `python -c "import gsplat, torch; assert torch.cuda.is_available()"`，失败时打印中文指引）；`bash server/scripts/run.sh [laptop|server]`。

- [ ] **Step 1: 写 GPU 端到端测试**：用 `make_capture` 生成 24 帧合成拍摄包（图像由 Task 4 的合成场景渲染，带深度），经真实 pipeline 跑 `iterations=1000`，断言 `state=="done"` 且 `scene.splat` 大小是 32 的整数倍并且 > 0。
- [ ] **Step 2:** 写 `install.sh`、`run.sh`，在笔记本的 WSL2 Ubuntu 中执行 install → 通过；`uv run pytest -m gpu -v` → 全部 PASS。
- [ ] **Step 3:** 写 `01-server-laptop.md`（中文，逐步，带截图位置说明）。内容包括：
  - 更新 NVIDIA 驱动，在 WSL 中用 `nvidia-smi` 验证；
  - 在 `%UserProfile%\.wslconfig` 设置 `[wsl2] networkingMode=mirrored`，让手机能访问 WSL 里的端口；
  - 放行 Windows 防火墙入站 TCP 8765（**由用户本人执行管理员命令**，文档给出命令和解释，执行者不代为修改系统或安全设置）；
  - 运行 install 和 run；
  - 在手机浏览器打开 `http://<笔记本IP>:8765/api/v1/health` 做验证。
- [ ] **Step 4:** 写 `02-server-lab.md`（相同脚本，`run.sh server`；提示需确认实验室网络是否允许手机访问 8765）。
- [ ] **Step 5:** 人工验证：用户按 01 文档操作后，手机浏览器能看到 health JSON。记录结果。
- [ ] **Step 6:** Commit `feat(server): one-command install/run and setup docs`。

---

### Task 8: Android 工程骨架 + 计算端设置

**Files:**
- Create: `android/` Gradle 工程（`settings.gradle.kts`、`build.gradle.kts`、`gradle/libs.versions.toml`、`app/build.gradle.kts`、`AndroidManifest.xml`）、`MainActivity.kt`、`nav/AppNav.kt`、`settings/ServerSettings.kt`、`net/ApiClient.kt`、`docs/setup/03-phone.md`
- Test: `android/app/src/test/java/dev/recon3d/net/ApiClientTest.kt`

**Interfaces:**
- Produces:
  - `ServerSettings(context)`：`val baseUrl: Flow<String?>`、`suspend fun setBaseUrl(url: String)`；输入如 `192.168.1.20:8765` 时自动补 `http://`。
  - `ApiClient(baseUrl: String, http: OkHttpClient)`：`suspend fun health(): Health`、`suspend fun upload(zip: File, onProgress: (Float)->Unit): String`（返回 job_id）、`suspend fun job(id: String): JobStatus`、`suspend fun downloadScene(id: String, dest: File)`；网络错误统一抛 `ApiException(messageZh: String)`。
  - 数据类 `Health`、`JobStatus` 字段与 `docs/api.md` 一致（kotlinx.serialization）。
  - Manifest：`CAMERA` 权限、`android.hardware.camera.ar` required、`com.google.ar.core` meta-data `required`、`usesCleartextTraffic=true`（局域网 HTTP）。

- [ ] **Step 0（用户操作）:** 用户安装 Android Studio（自带 JDK 和 SDK）。执行者按 `03-phone.md` 草稿指导用户完成；完成后 `android/gradlew --version` 可运行。
- [ ] **Step 1: 写失败测试**（MockWebServer）

```kotlin
@Test fun healthParses() = runTest {
    server.enqueue(MockResponse().setBody("""{"version":"0.1","gpu":"RTX 4060","profile":"laptop"}"""))
    assertEquals("laptop", client.health().profile)
}
@Test fun connectionRefusedGivesChineseMessage() = runTest {
    server.shutdown()
    val e = assertThrows(ApiException::class.java) { runBlocking { client.health() } }
    assertTrue(e.messageZh.contains("连不上计算端"))
}
@Test fun uploadReturnsJobId() = runTest {
    server.enqueue(MockResponse().setResponseCode(201).setBody("""{"job_id":"abc"}"""))
    assertEquals("abc", client.upload(tmpZip(1024)) {})
    assertTrue(server.takeRequest().body.readUtf8().contains("name=\"capture\""))
}
```

- [ ] **Step 2:** `./gradlew :app:testDebugUnitTest` → FAIL。
- [ ] **Step 3:** 实现。设置页包含：地址输入框、"测试连接"按钮；成功显示"已连接：RTX 4060（笔记本档）"，失败显示中文错误。
- [ ] **Step 4:** 测试 PASS；`./gradlew :app:assembleDebug` 成功；通过 USB 装到 Pixel，"测试连接"能连上 Task 7 的服务端。
- [ ] **Step 5:** 完成 `03-phone.md`（开启开发者模式、USB 调试、在 Android Studio 里点运行）。
- [ ] **Step 6:** Commit `feat(android): project skeleton, server settings, API client`。

---

### Task 9: 拍摄引导状态机（纯 Kotlin）

**Files:**
- Create: `capture/PoseMath.kt`, `capture/CaptureGuide.kt`
- Test: `app/src/test/java/dev/recon3d/capture/CaptureGuideTest.kt`, `PoseMathTest.kt`

**Interfaces:**
- Produces:
  - `PoseMath.yawPitchDeg(poseGl: FloatArray /*16, 列主序*/): Pair<Float, Float>`：前向 `f = R·(0,0,-1)`；`yaw = atan2(f.x, -f.z)`（度，-180..180，右转为正）；`pitch = asin(f.y)`（度，抬头为正）。
  - `enum class Stage { SPIN, SWEEP, ROAM, DONE }`；`enum class Arrow { LEFT, RIGHT, UP, DOWN, FORWARD_LEFT, FORWARD_RIGHT, NONE }`
  - `data class GuideUi(stage: Stage, stageProgress: Float, arrow: Arrow, hint: String, subHint: String, warning: String?, canFinish: Boolean)`
  - `class CaptureGuide { fun onFrame(tNs: Long, poseGl: FloatArray, tracking: Boolean): GuideUi; fun skipStage() }`
  - 规则：
    - **SPIN**：36 个 10° 偏航格，pitch 在 ±25° 内才计入；覆盖 ≥ 33 格完成。箭头 RIGHT；如果最近未覆盖的格在左边则 LEFT。提示文字："站在房间中央，慢慢向右转" / "保持手机与胸口同高"。
    - **SWEEP**：进度 = (到达过 pitch ≥ +45° ? 0.5 : 0) + (到达过 ≤ −45° ? 0.5 : 0)。先 UP 后 DOWN。提示："慢慢抬起，拍天花板" / "然后向下扫到地面"。
    - **ROAM**：起点之外的位置格（0.5 m 网格）× 8 个偏航格，累计覆盖 ≥ 24 个组合，并且累计位移 ≥ 1.5 m 时完成。箭头指向覆盖最少的偏航扇区（相对当前朝向转换为 FORWARD_LEFT / FORWARD_RIGHT / LEFT / RIGHT）。提示："向箭头方向走两步" / "还有角落没拍到"。
    - `warning`：角速度 > 60°/s →"太快了，慢一点"；`tracking=false` →"跟丢了，请回到刚才的位置"（此时不更新覆盖）。
    - `canFinish`：SPIN 完成之后为 true；SPIN 未完成时为 false，`hint` 解释"先完成原地转圈"。
    - `skipStage()`：只允许跳过 SWEEP 和 ROAM。

- [ ] **Step 1: 写失败测试**

```kotlin
@Test fun identityPoseLooksForward() { val (y, p) = PoseMath.yawPitchDeg(identity()); assertEquals(0f, y, 1e-3f); assertEquals(0f, p, 1e-3f) }
@Test fun fullSpinCompletesStage() { val g = CaptureGuide(); spin(g, 0..355 step 5); assertEquals(Stage.SWEEP, g.last.stage) }
@Test fun cannotFinishBeforeSpin() { val g = CaptureGuide(); spin(g, 0..90 step 5); assertFalse(g.last.canFinish); assertTrue(g.last.hint.contains("先完成原地转圈")) }
@Test fun trackingLostDoesNotAddCoverage() { val g = CaptureGuide(); spin(g, 0..355 step 5, tracking = false); assertEquals(0f, g.last.stageProgress) }
@Test fun fastRotationWarns() { val g = CaptureGuide(); g.onFrame(0, yaw(0f), true); val ui = g.onFrame(100_000_000, yaw(20f), true); assertEquals("太快了，慢一点", ui.warning) }
@Test fun sweepNeedsUpAndDown() { val g = afterSpin(); pitchTo(g, 50f); assertEquals(0.5f, g.last.stageProgress); assertEquals(Arrow.DOWN, g.last.arrow) }
```

- [ ] **Step 2:** 跑 → FAIL。
- [ ] **Step 3:** 实现（无 Android 依赖，便于单测）。
- [ ] **Step 4:** 跑 → PASS。
- [ ] **Step 5:** Commit `feat(android): three-stage capture guide`。

---

### Task 10: 关键帧选择 + 拍摄包写入

**Files:**
- Create: `capture/KeyframeSelector.kt`, `capture/CapturePackageWriter.kt`
- Test: `KeyframeSelectorTest.kt`, `CapturePackageWriterTest.kt`

**Interfaces:**
- Consumes: `PoseMath`（Task 9）
- Produces:
  - `KeyframeSelector(minIntervalNs = 200_000_000, minTranslationM = 0.05f, minRotationDeg = 5f)`：`fun shouldKeep(tNs, poseGl, tracking): Boolean`。`tracking=false` 一律 false；第一帧保留；之后要求距离上一个保留帧的时间 ≥ 间隔，并且平移或旋转超过阈值。
  - `CapturePackageWriter(dir: File, roomId: String, captureId: String, imageW, imageH, depthW?, depthH?)`：`fun addFrame(tNs, jpeg: ByteArray, depthMm: ShortArray?, poseGl: FloatArray, fx, fy, cx, cy)`（写 `images/`、`depth/`，追加一行 `frames.jsonl`）；`fun finish(): File`（写 `manifest.json`、打包 zip、返回 zip 文件）。格式严格按 `docs/capture-format.md`。
  - 深度 PNG 用 16 位灰度。Android `Bitmap` 不支持写 16 位 PNG，用一个最小的 16 位 PNG 编码器（`java.util.zip.Deflater` + CRC32，约 60 行），放在 `capture/Png16.kt`。

- [ ] **Step 1: 写失败测试**

```kotlin
@Test fun dropsFramesWhileNotTracking() { assertFalse(sel.shouldKeep(0, identity(), tracking = false)) }
@Test fun keepsFirstThenNeedsMotion() { assertTrue(sel.shouldKeep(0, identity(), true)); assertFalse(sel.shouldKeep(300_000_000, identity(), true)); assertTrue(sel.shouldKeep(600_000_000, translate(0.1f), true)) }
@Test fun packageMatchesFormatV1() {
    val zip = writer.apply { addFrame(1, jpeg(), depth(4, 3), identity(), 500f, 500f, 320f, 240f) }.finish()
    val m = readManifest(zip); assertEquals(1, m.formatVersion); assertEquals(1, m.frameCount)
    assertTrue(zipEntries(zip).containsAll(listOf("manifest.json", "frames.jsonl", "images/000001.jpg", "depth/000001.png")))
}
@Test fun png16RoundTrips() { val px = shortArrayOf(0, 1000, 65535.toShort(), 2000); assertArrayEquals(px, decodePng16(Png16.encode(2, 2, px))) }
```

- [ ] **Step 2:** 跑 → FAIL。
- [ ] **Step 3:** 实现。
- [ ] **Step 4:** 跑 → PASS。另外把一个 Kotlin 生成的拍摄包复制到 `server/tests/fixtures/android_v1.zip`，在 `server/tests/test_capture_format.py` 加 `test_loads_android_package`，作为跨语言契约测试。
- [ ] **Step 5:** Commit `feat(android): keyframe selection and capture package writer`。

---

### Task 11: AR 拍摄界面（风格 A）

**Files:**
- Create: `capture/ar/ArSessionHelper.kt`, `capture/ar/BackgroundRenderer.kt`（+ 着色器 assets）, `capture/CaptureScreen.kt`
- Modify: `nav/AppNav.kt`

**Interfaces:**
- Consumes: `CaptureGuide`、`KeyframeSelector`、`CapturePackageWriter`
- Produces: `CaptureScreen(roomId: String, onFinished: (zip: File) -> Unit)`
  - `ArSessionHelper.configure(session)`：从 `session.getSupportedCameraConfigs(CameraConfigFilter(...).setDepthSensorUsage(...))` 中选 **CPU 图像分辨率最大**（长边 ≤ 1920）的配置，同时开启 `Config.DepthMode.AUTOMATIC`（不支持则关闭，深度写 null）、`FocusMode.AUTO`。
  - 每个 GL 帧：`frame.camera.trackingState`、`camera.pose`（不用 displayOrientedPose）、`camera.imageIntrinsics` → `CaptureGuide.onFrame`；`KeyframeSelector.shouldKeep` 为 true 时，`acquireCameraImage()`（YUV_420_888）和 `acquireDepthImage16Bits()`，在 GL 线程上只复制数据，YUV→JPEG（质量 90）和写盘放到单独的协程 IO 线程；队列积压 > 8 帧时丢弃新的关键帧，避免卡顿。
  - 覆盖层严格按草图 A：顶部三个阶段胶囊（当前阶段蓝色，完成的打 ✓）；顶部居中进度环加百分比；屏幕中央一个大箭头（按 `Arrow` 旋转）；底部主提示和副提示；`warning` 用醒目的黄色条显示；底部红色录制按钮；`canFinish` 为 true 时出现"完成拍摄"按钮；ROAM 阶段有"跳过"按钮。

- [ ] **Step 1:** 实现 `ArSessionHelper` 与 `BackgroundRenderer`（改自 ARCore 官方 `hello_ar_kotlin` 示例的 BackgroundRenderer，保留 Apache-2.0 版权头）。
- [ ] **Step 2:** 实现 `CaptureScreen`：Compose `AndroidView(GLSurfaceView)` 上面叠 Compose 覆盖层；处理相机权限请求、ARCore 未安装或需更新（`ArCoreApk.requestInstall`）、生命周期 pause/resume。
- [ ] **Step 3:** `./gradlew :app:assembleDebug` 成功并安装到 Pixel。
- [ ] **Step 4: 人工验证**（执行者请用户在手机上操作并反馈）：
  - 实时画面正常、方向正确；
  - 原地转一圈，进度环从 0 涨到 100%，进入"② 上下"；
  - 甩快手机时出现"太快了，慢一点"；
  - 用手捂住镜头，出现"跟丢了…"；
  - 完成后得到 zip，用 `adb pull` 取到电脑，`uv run python -c "from recon3d.capture.format import load_capture; ..."` 能加载，图片长边 ≥ 1280。
- [ ] **Step 5:** Commit `feat(android): AR capture screen with arrow guide (style A)`。

---

### Task 12: 房间列表 + 上传 + 进度

**Files:**
- Create: `rooms/RoomRepository.kt`, `rooms/RoomListScreen.kt`, `upload/UploadScreen.kt`
- Test: `RoomRepositoryTest.kt`, `UploadViewModelTest.kt`

**Interfaces:**
- Consumes: `ApiClient`（Task 8）、`CaptureScreen`（Task 11）
- Produces:
  - `Room(id, name, createdAt, captures: List<CaptureRecord>)`；`CaptureRecord(id, kind="model", zipPath, jobId: String?, state: "local"|"uploading"|"queued"|"training"|"exporting"|"done"|"failed", message: String?, scenePath: String?)`
  - `RoomRepository(filesDir)`：`fun rooms(): Flow<List<Room>>`、`createRoom(name)`、`updateCapture(roomId, record)`，存为 `filesDir/rooms.json`，原子写入。
  - `UploadViewModel(api, repo)`：`fun start(roomId, captureId)`：上传 → 每 3 秒轮询 → `done` 时下载 `scene.splat` 到 `filesDir/rooms/<roomId>/<captureId>/scene.splat`。任何 `ApiException` 都把状态设为 `failed`，`message=messageZh`，**保留 zip**；`retry()` 重新上传。
  - 界面：房间列表（新建房间 → 拍摄 → 上传）；每条拍摄显示状态和进度条；`done` 显示"查看 3D"；`failed` 显示原因和"重新上传"。

- [ ] **Step 1: 写失败测试**

```kotlin
@Test fun uploadFailureKeepsZipAndAllowsRetry() = runTest {
    api.failUploadOnce(ApiException("连不上计算端，请检查 Wi-Fi 和地址"))
    vm.start(room, cap); advanceUntilIdle()
    assertEquals("failed", repo.capture(room, cap).state); assertTrue(File(repo.capture(room, cap).zipPath).exists())
    vm.retry(); advanceUntilIdle(); assertEquals("done", repo.capture(room, cap).state)
}
@Test fun serverFailedJobShowsMessage() = runTest {
    api.jobResult = JobStatus("j", "failed", 0f, "显存不足，请降低分辨率或改用服务器")
    vm.start(room, cap); advanceUntilIdle()
    assertEquals("显存不足，请降低分辨率或改用服务器", repo.capture(room, cap).message)
}
@Test fun repositoryPersists() { repo.createRoom("卧室"); assertEquals("卧室", RoomRepository(dir).roomsNow().single().name) }
```

- [ ] **Step 2:** 跑 → FAIL。
- [ ] **Step 3:** 实现（`ApiClient` 抽一个接口 `Api`，测试里用假实现）。
- [ ] **Step 4:** 跑 → PASS；在真机上上传一次真实拍摄，笔记本上训练完成，手机显示"查看 3D"。
- [ ] **Step 5:** Commit `feat(android): rooms, upload and job progress`。

---

### Task 13: 手机端 3D 浏览

**Files:**
- Create: `viewer/ViewerScreen.kt`, `app/src/main/assets/viewer/index.html`, `app/src/main/assets/viewer/viewer.js`, `android/viewer-web/`（package.json + 打包脚本，产物拷进 assets）

**Interfaces:**
- Consumes: `CaptureRecord.scenePath`（Task 12）
- Produces:
  - `ViewerScreen(scenePath: String)`：WebView 通过 `WebViewAssetLoader` 从 `https://appassets.androidplatform.net/assets/viewer/index.html` 加载页面；场景文件通过 `/scene/scene.splat` 路径映射到本地文件（自定义 `PathHandler`），不开放任意文件访问。
  - `viewer.js` 暴露 JS API（第二期会用到，本期实现并手测）：`loadScene(url)`、`flyTo({x,y,z}, {x,y,z} lookAt, durationMs)`、`setCameraHome()`。坐标为 CV 世界坐标，viewer 内负责转换到 three.js（Y 轴取反）。
  - 交互：单指旋转、双指缩放和平移（three.js `OrbitControls`）；初始相机放在第一帧相机位置，朝向它的前方。Spark 和 three.js 用 npm 安装并钉死精确版本，打包进 assets，**不依赖 CDN**。
  - 本期明确不做（设计文档要求查看器最终支持，但留到后续）：按物体高亮和调暗其余部分（第二期）、半透明影子（第三期）、局域网自动发现计算端（本期只支持手填地址）。

- [ ] **Step 1:** 在 `android/viewer-web` 用 npm 安装 `@sparkjsdev/spark` 和 `three` 的精确版本，配 esbuild 打包到 `app/src/main/assets/viewer/`；`npm run build` 成功。
- [ ] **Step 2:** 实现 `index.html`、`viewer.js`、`ViewerScreen`。
- [ ] **Step 3: 人工验证：**
  - 打开 Task 12 得到的真实房间，场景能显示，方向正确（地面在下）；
  - 拖动流畅，主观上不卡；
  - 在 Chrome 远程调试控制台调用 `flyTo(...)`，镜头会平滑移动。
- [ ] **Step 4:** Commit `feat(android): on-device 3DGS viewer`。

---

### Task 14: 第一期验收

**Files:**
- Create: `docs/acceptance/phase1.md`, `README.md`（补全）

- [ ] **Step 1:** 写验收清单，直接引用设计文档第 6 节第一期的标准：约 15 m² 卧室，3 分钟内拍完；笔记本 30 分钟内重建完；手机浏览流畅；家具和桌面物品可辨认。每一项附测量方法（计时、截图）。
- [ ] **Step 2:** 和用户一起按清单实测一次，把实测数字填进文档。哪一项不达标就记成问题，不在本任务里修。
- [ ] **Step 3:** 服务端全部测试：`uv run pytest -v` 和 `uv run pytest -m gpu -v` 全绿；Android：`./gradlew :app:testDebugUnitTest` 全绿。
- [ ] **Step 4:** Commit `docs: phase 1 acceptance results`。
