# Recon3D 第一期补丁：减少 floater（Raw 深度 + 置信度、深度约束、位姿微调）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让房间重建里的悬空碎片（floater）明显减少：手机改采 ARCore Raw Depth + 置信度，服务端只用高置信深度做初始化，训练加鲁棒深度约束并微调相机位姿，拍摄引导把"太快"阈值降到 30°/s。

**Architecture:** 拍摄包升到 v2（新增 `conf/%06d.png` 和每帧 `depth_kind`），服务端 v1/v2 都收，统一成"每帧一张 0..1 置信度图"（v1 用深度梯度启发式生成）。初始点云按置信度和 4 m 范围过滤；训练用 gsplat `render_mode="RGB+ED"` 得到期望深度，在深度分辨率上做 Huber 深度损失（置信度加权），同时用一个 6-DoF 小量位姿修正模块微调每个相机。新增离线评估脚本（逐视角 PSNR + floater 分数）用 office 拍摄包做前后对比。

**Tech Stack:** Kotlin/ARCore 1.57（`acquireRawDepthImage16Bits`, `acquireRawDepthConfidenceImage`），Python 3.10 / torch 2.4.1+cu124 / gsplat 1.5.3，pytest（`gpu` 标记），JUnit。

**Spec:** `docs/superpowers/specs/2026-11-01-recon3d-design.md` §5.3（"用 ARCore 的深度或特征点云初始化高斯点"、"如果 ARCore 的位姿有漂移…可以在服务端加一步位姿优化"）。诊断依据：`server/scripts/diag_poses.py` 在 office 拍摄包上的结果——相邻关键帧位姿与图像一致到 0.6 px / 0.7°，10 帧外约 3 px；平滑深度图相邻帧互差中位数 4.7 cm、p90 20 cm，玻璃窗被估成 >6 m。

## Global Constraints

- 用户可见文字一律中文；代码注释英文（与现有代码一致）。
- 服务端 Python 3.10，依赖不新增（OpenCV 只用于诊断脚本，`uv run --with`）。
- 拍摄包 **v1 必须继续可用**（`server/tests/fixtures/android_v1.zip` 的测试不变）。
- 服务端测试：`cd server && UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d uv run pytest -q -rs`（含 GPU 测试，`skipped` 不算通过）。
- Android 测试：`cd android; $env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'; .\gradlew.bat :app:testDebugUnitTest :app:assembleDebug`。
- 提交信息末尾：`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。
- 不改 `.splat` 导出格式；世界坐标系仍是 ARCore 世界系（位姿微调是对每个相机的小修正，带 L2 正则，不整体漂移）。

## Review Focus

1. **v1 拍摄包**（无 `confidence`、无 `depth_kind`）必须照常训练——Task 1/2 的测试用 `android_v1.zip` 和 `make_capture(format_version=1)` 走完整 `build_training_set` + `init_points`。
2. **Raw depth 过期**：ARCore 的 raw depth 可能来自更早的一帧；时间戳不等于当前帧时必须回退到平滑深度并标 `depth_kind="smooth"`，不能把旧深度配给新位姿——Task 6 的测试覆盖"时间戳不一致 → smooth"。
3. **置信度图尺寸与深度图不一致**（理论上不会，但要防）：服务端跳过该帧的深度并记日志，不报错——Task 2 测试。
4. **深度分辨率不是图像的整数分之一**（例如 1600 宽的 server 档：1600/160 = 10 整除；但 1280/160 = 8 也整除；若用户换机型 depth 为 240×180 而图像 1920×1080 → 不整除）：深度损失对该帧跳过，训练不崩——Task 4 测试。
5. **位姿微调不能把场景整体搬走**：所有相机同时平移会让 ARCore 原点失效（查看器起始视角、第三期对齐都依赖它）。Task 4 测试：零参数等价于原位姿；正则项让 6 维参数的 L2 范数被惩罚。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `server/recon3d/capture/format.py`（改） | v1/v2 加载；`Frame.confidence`, `Frame.depth_kind` |
| `server/tests/fixtures/make_capture.py`（改） | 生成 v2 包（`confidences=`，`depth_kind=`） |
| `server/recon3d/recon/confidence.py`（新） | `depth_confidence(depth_m, conf_png) -> np.ndarray`：统一的 0..1 置信度 |
| `server/recon3d/recon/dataset.py`（改） | `TrainingSet.confs`；加载 conf PNG 或调用启发式 |
| `server/recon3d/recon/init_points.py`（改） | 置信度 + 4 m 过滤 |
| `server/recon3d/recon/pose_opt.py`（新） | `PoseRefine`：每相机 6-DoF 小量修正 |
| `server/recon3d/recon/trainer.py`（改） | 深度损失、位姿微调 |
| `server/recon3d/config.py`（改） | `Profile.depth_lambda`, `Profile.pose_lr` |
| `server/scripts/eval_scene.py`（新） | 逐视角 PSNR + floater 分数 |
| `android/.../capture/ar/CaptureRenderer.kt`, `CapturePipeline.kt`（改） | 采 raw depth + confidence，时间戳校验 |
| `android/.../capture/CapturePackageWriter.kt`, `Png16.kt`（改） | 写 `conf/*.png`（8 位）、`depth_kind`、`format_version: 2` |
| `android/.../capture/CaptureGuide.kt`（改） | `MAX_DEG_PER_S = 30.0` |
| `docs/capture-format.md`, `docs/acceptance/phase1.md`（改） | v2 说明；对比结果 |

---

### Task 1: 拍摄包格式 v2（服务端加载器 + 测试生成器 + 文档）

**Files:**
- Modify: `server/recon3d/capture/format.py`
- Modify: `server/tests/fixtures/make_capture.py`
- Modify: `docs/capture-format.md`
- Test: `server/tests/test_capture_format.py`

**Interfaces:**
- Produces: `Frame.confidence: str | None = None`（相对路径，如 `conf/000001.png`，8 位灰度 PNG，0..255，255 最可信）；`Frame.depth_kind: str = "smooth"`（`"raw"` | `"smooth"`）；`SUPPORTED_VERSIONS = (1, 2)`；`make_capture(..., format_version=2, confidences=list[np.ndarray uint8 HxW] | None, depth_kind="raw")`。
- v2 规则：`format_version == 2`；`confidence` 可为 null；有 `confidence` 时文件必须存在且与深度同尺寸（尺寸校验在 Task 2，这里只校验存在和路径合法）；`depth_kind` 缺省 `"smooth"`，只允许两个值。

- [ ] **Step 1: 写失败的测试**（追加到 `server/tests/test_capture_format.py`）

```python
def test_v2_with_confidence_loads(tmp_path):
    conf = [np.full((6, 8), 200, np.uint8)] * 2
    z = make_capture(tmp_path, n_frames=2, format_version=2, confidences=conf, depth_kind="raw",
                     depths=[np.full((6, 8), 1500, np.uint16)] * 2)
    cap = load_capture(z, tmp_path / "w")
    assert cap.manifest.format_version == 2
    assert cap.frames[0].confidence == "conf/000001.png"
    assert cap.frames[0].depth_kind == "raw"
    assert (cap.root / "conf/000001.png").is_file()


def test_v2_without_confidence_defaults_smooth(tmp_path):
    z = make_capture(tmp_path, n_frames=1, format_version=2)
    fr = load_capture(z, tmp_path / "w").frames[0]
    assert fr.confidence is None and fr.depth_kind == "smooth"


def test_v2_missing_confidence_file_is_error(tmp_path):
    z = make_capture(tmp_path, n_frames=1, format_version=2, confidences=[np.zeros((6, 8), np.uint8)],
                     depths=[np.zeros((6, 8), np.uint16)], drop_confidence=1)
    with pytest.raises(CaptureFormatError, match="缺少置信度图"):
        load_capture(z, tmp_path / "w")


def test_bad_depth_kind_is_error(tmp_path):
    z = make_capture(tmp_path, n_frames=1, format_version=2, depth_kind="fuzzy")
    with pytest.raises(CaptureFormatError, match="frames.jsonl 格式错误"):
        load_capture(z, tmp_path / "w")


def test_v3_rejected(tmp_path):
    with pytest.raises(CaptureFormatError, match="不支持的拍摄包版本: 3"):
        load_capture(make_capture(tmp_path, format_version=3), tmp_path / "w")
```

（现有 `test_unsupported_version` 若断言的是 version 2 被拒，改为 3。）

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest -q tests/test_capture_format.py`
Expected: 新测试 FAIL（`make_capture() got an unexpected keyword argument` / 属性不存在）。

- [ ] **Step 3: 实现**

`format.py`：`SUPPORTED_VERSIONS = (1, 2)`；`Frame` 增加两个字段，`depth_kind` 用 `Literal["raw", "smooth"]`（pydantic 校验失败即落入现有 `ValidationError → "frames.jsonl 格式错误"`）；版本检查改为 `version not in SUPPORTED_VERSIONS`；文件检查循环对 `fr.confidence` 做 `_inside` 校验并在缺失时抛 `CaptureFormatError(f"缺少置信度图: {rel}")`。
`make_capture.py`：新增参数 `confidences: list[np.ndarray] | None = None`, `depth_kind: str | None = None`, `drop_confidence: int | None = None`；`format_version == 2` 时每帧写 `"depth_kind"`（给定值，否则 `"raw"` if confidences else `"smooth"`）和 `"confidence"`（有 confidences 时 `conf/{i:06d}.png`，用 `Image.fromarray(arr, "L")` 写 PNG；`drop_confidence == i` 时只写字段不写文件）。
`docs/capture-format.md`：标题改"拍摄包格式 v1 / v2"，新增 v2 一节：`format_version: 2`；每帧 `depth_kind`（`"raw"` = ARCore Raw Depth，未平滑未填洞；`"smooth"` = `acquireDepthImage16Bits` 的平滑深度）；`confidence`：8 位灰度 PNG，与深度同尺寸，0..255，255 最可信，无 raw depth 的帧为 null；服务端对 v1 包和 `confidence` 为 null 的帧用启发式置信度（见 Task 2）。

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest -q tests/test_capture_format.py tests/test_pipeline.py`
Expected: 全部 PASS（含 `android_v1.zip` 的测试）。

- [ ] **Step 5: Commit**

```bash
git add server/recon3d/capture/format.py server/tests/fixtures/make_capture.py server/tests/test_capture_format.py docs/capture-format.md
git commit -m "feat(format): capture format v2 with raw-depth confidence maps"
```

---

### Task 2: 统一置信度（conf PNG 或启发式）进入 TrainingSet

**Files:**
- Create: `server/recon3d/recon/confidence.py`
- Modify: `server/recon3d/recon/dataset.py`
- Test: `server/tests/test_confidence.py`（新）, `server/tests/test_dataset.py`

**Interfaces:**
- Produces: `depth_confidence(depth_m: np.ndarray[float32, HxW], conf_u8: np.ndarray[uint8, HxW] | None) -> np.ndarray[float32, HxW]`，值域 0..1：
  - 无效深度（`<= 0`、非有限、`> MAX_DEPTH_M = 4.0`）→ 0；
  - `conf_u8` 给定 → `conf_u8 / 255`（再乘上面的有效掩码）；
  - 否则启发式：`1.0` 减去深度边缘惩罚：与 4 邻域深度差的最大值 `g`（米），`g >= EDGE_M = 0.10` → 0，否则 `1 - g / EDGE_M`；边界像素按最近邻填充处理。
- `TrainingSet.confs: list[torch.Tensor | None]`（每帧 `[dh, dw]` float32，无深度的帧为 None）。`build_training_set` 对有深度的帧调用 `depth_confidence`；有 `fr.confidence` 时读 PNG 为 uint8；若 conf 尺寸 ≠ 深度尺寸：`log.warning`，该帧 `depths[i] = None, confs[i] = None`（与现有宽高比跳过同样处理）。
- Consumes: Task 1 的 `Frame.confidence`。

- [ ] **Step 1: 写失败的测试**

`tests/test_confidence.py`：

```python
import numpy as np
from recon3d.recon.confidence import depth_confidence, MAX_DEPTH_M, EDGE_M

def test_conf_png_scales_and_masks_invalid():
    d = np.array([[1.0, 0.0], [5.0, 2.0]], np.float32)
    c = np.array([[255, 255], [255, 51]], np.uint8)
    out = depth_confidence(d, c)
    assert out[0, 0] == 1.0 and out[0, 1] == 0.0 and out[1, 0] == 0.0
    assert abs(out[1, 1] - 0.2) < 1e-6

def test_heuristic_penalises_depth_edges():
    d = np.full((5, 5), 2.0, np.float32)
    d[:, 3:] = 2.5  # 50 cm step between columns 2 and 3
    out = depth_confidence(d, None)
    assert out[2, 0] == 1.0          # flat region
    assert out[2, 2] == 0.0 and out[2, 3] == 0.0  # both sides of the step
    d2 = np.full((5, 5), 2.0, np.float32); d2[:, 3:] = 2.05
    assert abs(depth_confidence(d2, None)[2, 2] - 0.5) < 1e-6

def test_heuristic_far_is_zero():
    d = np.full((3, 3), MAX_DEPTH_M + 0.5, np.float32)
    assert depth_confidence(d, None).max() == 0.0
```

`tests/test_dataset.py` 追加：

```python
def test_training_set_confs_from_png(tmp_path):
    conf = [np.full((6, 8), 128, np.uint8)]
    z = make_capture(tmp_path, n_frames=1, format_version=2, confidences=conf,
                     depths=[np.full((6, 8), 1500, np.uint16)])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    assert ts.confs[0].shape == (6, 8)
    assert abs(float(ts.confs[0][0, 0]) - 128 / 255) < 1e-6

def test_training_set_confs_heuristic_for_v1(tmp_path):
    z = make_capture(tmp_path, n_frames=1, depths=[np.full((6, 8), 1500, np.uint16)])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    assert float(ts.confs[0].min()) == 1.0  # flat depth, no edges

def test_conf_size_mismatch_skips_depth(tmp_path, caplog):
    conf = [np.full((3, 4), 255, np.uint8)]
    z = make_capture(tmp_path, n_frames=1, format_version=2, confidences=conf,
                     depths=[np.full((6, 8), 1500, np.uint16)])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    assert ts.depths[0] is None and ts.confs[0] is None
    assert "confidence" in caplog.text
```

- [ ] **Step 2: 运行，确认失败**

Run: `uv run pytest -q tests/test_confidence.py tests/test_dataset.py`
Expected: ImportError / AttributeError。

- [ ] **Step 3: 实现** `confidence.py` 与 `dataset.py` 改动（见 Interfaces；`TrainingSet` 的 `confs` 字段默认 `field(default_factory=list)` 以免破坏现有构造处——检查 `tests/fixtures/synthetic.py` 若直接构造 `TrainingSet`，给它补 `confs=[None]*N` 或全 1）。

- [ ] **Step 4: 运行，确认通过**

Run: `uv run pytest -q tests/test_confidence.py tests/test_dataset.py tests/test_init_points.py tests/test_trainer.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add server/recon3d/recon/confidence.py server/recon3d/recon/dataset.py server/tests/test_confidence.py server/tests/test_dataset.py server/tests/fixtures/synthetic.py
git commit -m "feat(recon): per-frame depth confidence (conf png or edge heuristic)"
```

---

### Task 3: 初始点云按置信度过滤

**Files:**
- Modify: `server/recon3d/recon/init_points.py`
- Test: `server/tests/test_init_points.py`

**Interfaces:**
- Consumes: `TrainingSet.confs`。
- Produces: `init_points(ts, voxel_size=0.02, min_points=5000, seed=0, conf_min: float = CONF_MIN)`，`CONF_MIN = 0.5`；像素保留条件改为 `confs[i] >= conf_min`（`confs[i]` 已含 0.3..4 m 范围掩码，所以删掉本文件的 `MAX_DEPTH_M` 判断，改为引用 `confidence.MAX_DEPTH_M` 仅用于文档/断言）。`confs[i] is None` 的帧跳过。

- [ ] **Step 1: 写失败的测试**（追加）

```python
def test_init_points_drops_low_confidence(tmp_path):
    depth = np.full((6, 8), 1500, np.uint16)
    conf = np.full((6, 8), 255, np.uint8); conf[:, :4] = 0
    z = make_capture(tmp_path, n_frames=1, format_version=2, depths=[depth], confidences=[conf])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    xyz, _ = init_points(ts, voxel_size=0.001, min_points=0)
    # right half only: x in camera frame >= 0 for u >= cx (cx = dw/2 scaled)
    assert len(xyz) == 6 * 4
    assert (xyz[:, 0] >= -1e-6).all()

def test_init_points_v1_drops_far_and_edges(tmp_path):
    depth = np.full((6, 8), 1500, np.uint16); depth[:, 6:] = 5000  # 5 m: beyond 4 m, plus an edge at col 5|6
    z = make_capture(tmp_path, n_frames=1, depths=[depth])
    ts = build_training_set(load_capture(z, tmp_path / "w"), PROFILES["laptop"])
    xyz, _ = init_points(ts, voxel_size=0.001, min_points=0)
    assert len(xyz) == 6 * 5  # columns 0..4 survive
```

- [ ] **Step 2: 运行，确认失败** — `uv run pytest -q tests/test_init_points.py`（计数不符）。

- [ ] **Step 3: 实现**（按 Interfaces）。

- [ ] **Step 4: 运行，确认通过** — `uv run pytest -q tests/test_init_points.py tests/test_trainer.py`。

- [ ] **Step 5: Commit** — `git commit -m "feat(recon): confidence-filtered initial point cloud (<= 4 m)"`

---

### Task 4: 深度损失 + 相机位姿微调

**Files:**
- Create: `server/recon3d/recon/pose_opt.py`
- Modify: `server/recon3d/recon/trainer.py`, `server/recon3d/config.py`
- Test: `server/tests/test_pose_opt.py`（新）, `server/tests/test_trainer.py`

**Interfaces:**
- `Profile` 新增 `depth_lambda: float = 0.05`, `pose_lr: float = 1e-4`（两档相同；`pose_lr = 0` 关闭位姿微调，`depth_lambda = 0` 关闭深度损失）。
- `pose_opt.py`：
  ```python
  class PoseRefine(torch.nn.Module):
      def __init__(self, n_cams: int, device: str): ...  # self.delta = nn.Parameter(zeros(n_cams, 6))  [omega(3), t(3)]
      def forward(self, viewmats: Tensor[M,4,4], idx: Tensor[M] long) -> Tensor[M,4,4]:
          """Left-multiplies each viewmat by exp(omega, t) (camera-frame correction)."""
      def reg(self) -> Tensor: """mean of ||delta||^2"""
  def se3_exp(omega: Tensor[M,3], t: Tensor[M,3]) -> Tensor[M,4,4]  # Rodrigues; omega=0 -> identity
  ```
  正则权重 `POSE_REG = 1e-2`，加到总损失：`loss += POSE_REG * pose.reg()`。优化器 Adam(lr=profile.pose_lr)，不衰减。微调从 `refine_start_iter`（=min(500, iters//10)）开始生效（之前梯度不回传到 delta），避免初始点云还没稳定时位姿乱动。
- `trainer.py`：
  - `_rasterize(..., render_mode: str = "RGB")`；训练时 `render_mode="RGB+ED"`，`img = out[..., :3]`, `depth = out[..., 3]`（`[1,H,W]`，期望深度，米）。
  - 深度损失（新函数，可单测）：
    ```python
    def depth_loss(rendered: Tensor[1,H,W], target: Tensor[dh,dw], conf: Tensor[dh,dw], delta: float = 0.1) -> Tensor | None:
        """Huber(|pooled(rendered) - target|, delta) weighted by conf, mean over conf>0; None if H % dh or W % dw != 0."""
    ```
    pooled 用 `F.avg_pool2d(kernel=(H//dh, W//dw))`；`target == 0` 的像素权重为 0（conf 已保证）。
  - 总损失：`(1-λs)·L1 + λs·(1-SSIM) + depth_lambda · depth_loss + POSE_REG · pose.reg()`（`depth_loss` 为 None 或该帧 `confs[i] is None` 时不加）。
  - 返回值不变（`Splats`）；另外在 `train()` 结尾 `log.info` 位姿修正的统计：旋转中位数/最大（度）、平移中位数/最大（厘米），便于验收。
- `render()` 不变（导出/评估仍用原始 ARCore 位姿；微调量很小，且世界系不变）。

- [ ] **Step 1: 写失败的测试**

`tests/test_pose_opt.py`：

```python
import math, torch
from recon3d.recon.pose_opt import PoseRefine, se3_exp

def test_zero_delta_is_identity():
    T = se3_exp(torch.zeros(2, 3), torch.zeros(2, 3))
    assert torch.allclose(T, torch.eye(4).expand(2, 4, 4))

def test_rotation_about_z_90deg():
    T = se3_exp(torch.tensor([[0.0, 0.0, math.pi / 2]]), torch.zeros(1, 3))
    p = T[0, :3, :3] @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(p, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)

def test_module_applies_per_camera_and_reg():
    m = PoseRefine(3, "cpu")
    with torch.no_grad():
        m.delta[1, 3] = 0.1  # 10 cm x-translation on camera 1 only
    v = torch.eye(4).expand(3, 4, 4).clone()
    out = m(v, torch.arange(3))
    assert torch.allclose(out[0], torch.eye(4)) and torch.allclose(out[2], torch.eye(4))
    assert abs(float(out[1, 0, 3]) - 0.1) < 1e-6
    assert abs(float(m.reg()) - (0.1 ** 2) / 3) < 1e-9
```

`tests/test_trainer.py` 追加（CPU）：

```python
def test_depth_loss_pools_and_weights():
    from recon3d.recon.trainer import depth_loss
    rendered = torch.full((1, 6, 8), 2.0)
    target = torch.full((3, 4), 2.0); target[0, 0] = 3.0   # 1 m error at one cell
    conf = torch.ones(3, 4); conf[0, 1] = 0.0
    l = depth_loss(rendered, target, conf, delta=0.1)
    # Huber(1.0, 0.1) = 0.1*(1.0 - 0.05) = 0.095 ; mean over the 11 cells with conf>0
    assert abs(float(l) - 0.095 / 11) < 1e-6

def test_depth_loss_none_when_not_divisible():
    from recon3d.recon.trainer import depth_loss
    assert depth_loss(torch.zeros(1, 7, 8), torch.zeros(3, 4), torch.ones(3, 4)) is None
```

GPU 测试（`@pytest.mark.gpu`，在 `test_trainer.py` 现有 GPU 用例旁）：`train()` 在 `tests/fixtures/synthetic.py` 的合成集上以 `Profile(..., iterations=300, depth_lambda=0.05, pose_lr=1e-4)` 跑完，不报错，返回的 `Splats.means` 数量 > 0；再以 `depth_lambda=0, pose_lr=0` 跑一次，同样通过（旧路径）。

- [ ] **Step 2: 运行，确认失败** — `uv run pytest -q tests/test_pose_opt.py tests/test_trainer.py`。

- [ ] **Step 3: 实现**（按 Interfaces；`se3_exp` 用 Rodrigues 公式，小角度分支 `theta < 1e-6` 用一阶近似）。

- [ ] **Step 4: 运行，确认通过** — `uv run pytest -q -rs tests/test_pose_opt.py tests/test_trainer.py tests/test_e2e_gpu.py`，GPU 用例不得 skipped。

- [ ] **Step 5: Commit** — `git commit -m "feat(recon): confidence-weighted depth loss and per-camera pose refinement"`

---

### Task 5: 离线评估脚本 + office 拍摄包前后对比

**Files:**
- Create: `server/scripts/eval_scene.py`
- Modify: `docs/acceptance/phase1.md`（第 5 节追加对比数据）

**Interfaces:**
- `uv run python server/scripts/eval_scene.py <job_dir> [--profile laptop]`：`<job_dir>` 含 `capture.zip`（或已解包的 `frames.jsonl`）和 `scene.ply`；加载 PLY 为 `Splats`（读 `export_splat.export_ply` 写出的字段；若没有现成 loader，在 `export_splat.py` 加 `load_ply(path) -> Splats`，带单测：export 后 load 回来与原 Splats `allclose`），用 `trainer.render` 渲染全部训练视角，输出：
  - 逐视角 PSNR 的中位数 / 最小值 / 最差 5 帧索引；
  - **floater 分数**：不透明度 > 0.5 的高斯中，离"置信度 ≥ 0.5 的初始点云（Task 3 的 `init_points`）"最近距离 > 10 cm 的比例（cKDTree）；
  - 高斯总数。
- 训练命令：`uv run recon3d-server` 不方便离线跑单个包，所以脚本再提供 `--train`：对 `capture.zip` 用 `jobs/pipeline.py` 同样的流程训练到 `<job_dir>/retrain/`（复用 `pipeline.run` 或等价函数，不复制训练代码）。

- [ ] **Step 1: 写失败的测试**（`tests/test_export_splat.py` 追加 `test_ply_roundtrip`：export_ply → load_ply → 各字段 allclose）。
- [ ] **Step 2: 运行，确认失败。**
- [ ] **Step 3: 实现** `load_ply` 与脚本。
- [ ] **Step 4: 运行，确认通过** — `uv run pytest -q tests/test_export_splat.py`；再手动：
  - `uv run python server/scripts/eval_scene.py server/data/jobs/cfd1968452b2`（改动前的模型，基线）
  - `uv run python server/scripts/eval_scene.py server/data/jobs/cfd1968452b2 --train`（新代码重训 + 评估）
  把两组数字（PSNR 中位/最小、floater 分数、高斯数、训练用时）写进 `docs/acceptance/phase1.md` 第 5 节第 1 条之后的新小节"floater 对比（2026-11-xx）"。期望：floater 分数下降、PSNR 不降；若 PSNR 中位数下降超过 0.5 dB，记录并在任务报告里说明（不在本任务调参）。
- [ ] **Step 5: Commit** — `git commit -m "tools: offline scene evaluation (per-view PSNR, floater score) with before/after numbers"`

---

### Task 6: Android 采集 Raw Depth + 置信度，写 v2 包

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/capture/ar/CapturePipeline.kt`（`DepthBytes`, 透传）
- Modify: `android/app/src/main/java/dev/recon3d/capture/ar/CaptureRenderer.kt`（`acquireDepth`）
- Modify: `android/app/src/main/java/dev/recon3d/capture/CapturePackageWriter.kt`, `Png16.kt`（新增 `Png8`）
- Modify: `android/app/src/main/java/dev/recon3d/capture/ImageConvert.kt`（置信度平面拷贝）
- Test: `CapturePackageWriterTest.kt`, `CapturePipelineTest.kt`, `Png16Test.kt`

**Interfaces:**
- `DepthBytes(data, width, height, rowStrideBytes, intrinsics, kind: String = "smooth", confidence: ByteArray? = null, confRowStride: Int = 0)`。
- `CaptureRenderer.acquireDepth(frame, camera)`：先 `frame.acquireRawDepthImage16Bits()`；若 `raw.timestamp == frame.timestamp` 则同时 `frame.acquireRawDepthConfidenceImage()`（Y8，`planes[0]`），返回 `kind="raw"` 带置信度；若时间戳不等或抛 `NotYetAvailableException`，回退到现有 `acquireDepthImage16Bits()`，`kind="smooth"`，`confidence=null`。两条路径都用 `textureIntrinsics` 缩放得到 `intrinsics`。时间戳校验逻辑抽成纯函数 `fun rawDepthIsFresh(rawTs: Long, frameTs: Long): Boolean = rawTs == frameTs`（放 `CapturePipeline.kt` 顶层，可单测）。
- `ImageConvert.plane8(data: ByteArray, w: Int, h: Int, rowStride: Int): ByteArray`：去掉行填充，得到 `w*h` 字节。
- `Png8.encode(width, height, pixels: ByteArray): ByteArray`（8 位灰度，放在 `Png16.kt` 同文件或新文件 `Png8.kt`，复用 deflate/chunk 代码——把 `Png16` 的私有工具抽成 `internal object PngWriter`）。
- `CapturePackageWriter.addFrame(..., depthIntrinsics, depthKind: String = "smooth", confidence: ByteArray? = null)`：有深度时写 `"depth_kind"`；`confidence != null` 时写 `conf/%06d.png` 和 `"confidence"` 字段，否则 `"confidence": null`；`finish()` 的 `format_version` 改为 2。`init` 里 `File(dir, "conf").mkdirs()`。
- `CapturePipeline`：`w.addFrame(..., depthKind = d?.kind ?: "smooth", confidence = d?.confidence?.let { ImageConvert.plane8(it, d.width, d.height, d.confRowStride) })`。
- 固件：Kotlin 测试生成 `server/tests/fixtures/android_v2.zip`（仿照现有 `android_v1.zip` 的生成测试，2 帧，含 conf），Python 侧 `test_pipeline.py` 加一条用它跑通 `load_capture + build_training_set`（depth_kind raw、confs 来自 PNG）。

- [ ] **Step 1: 写失败的测试**
  - `Png16Test`：`Png8.encode(2, 1, byteArrayOf(0, 255.toByte()))` 用 `java.util.zip.Inflater`/手工解析 IHDR 断言 bit depth 8、color type 0、像素 `[0, 255]`（仿照现有 Png16 测试的解析方式）。
  - `CapturePackageWriterTest`：`addFrame(..., depthKind = "raw", confidence = ByteArray(dw*dh){200.toByte()})` → frames.jsonl 行含 `"depth_kind":"raw"`、`"confidence":"conf/000001.png"`，文件存在；`addFrame(... depthKind="smooth", confidence=null)` → `"confidence":null`；`finish()` 的 manifest `format_version == 2`。
  - `CapturePipelineTest`：`rawDepthIsFresh(5, 5)` true、`(4, 5)` false；一个 `DepthBytes(kind="raw", confidence=…)` 的 job 经 pipeline 写出 conf 文件（现有 pipeline 测试的假 writer/真 writer 方式沿用）。
  - 固件生成测试：产出 `android_v2.zip` 到 `server/tests/fixtures/`（与 v1 的做法一致，固定 ZipEntry 时间保证可复现）。
- [ ] **Step 2: 运行，确认失败** — `.\gradlew.bat :app:testDebugUnitTest`。
- [ ] **Step 3: 实现**（按 Interfaces）。
- [ ] **Step 4: 运行，确认通过** — Android 全部单元测试 + `assembleDebug`；Python：`uv run pytest -q tests/test_pipeline.py tests/test_capture_format.py`。
- [ ] **Step 5: Commit** — `git commit -m "feat(capture): ARCore raw depth + confidence, capture format v2"`

---

### Task 7: 拍摄引导"太快"阈值 30°/s

**Files:**
- Modify: `android/app/src/main/java/dev/recon3d/capture/CaptureGuide.kt:31`
- Test: `CaptureGuideTest.kt`

- [ ] **Step 1: 写失败的测试**：两帧相隔 0.1 s、偏航差 4°（40°/s）→ `warning == "太快了，慢一点"`；偏航差 2°（20°/s）→ `warning == null`。
- [ ] **Step 2: 运行，确认失败**（40°/s 现在不报警）。
- [ ] **Step 3: 实现**：`MAX_DEG_PER_S = 30.0`；若现有测试用 50°/s 作为"不报警"样例，改为 20°/s。
- [ ] **Step 4: 运行，确认通过**。
- [ ] **Step 5: Commit** — `git commit -m "feat(guide): warn above 30 deg/s to limit motion blur"`

---

### Task 8: 文档收尾与真机清单

**Files:**
- Modify: `docs/acceptance/phase1.md`（第 4 节第 5 项加 raw depth 检查：logcat `CaptureRenderer` 应打印 `depth ready` 之外的 `raw depth: fresh N / stale M` 计数——Task 6 实现时在 `logIntrinsicsOnce` 旁加一条每次拍摄结束时的计数日志；拍摄包里 `conf/` 数量应 ≈ 帧数的大半），第 5 节标记第 1、3（部分）条的进展。
- Modify: `docs/setup/03-phone.md`：无改动则不动。
- Modify: `README.md`：一句话提到 v2 包与 v1 兼容。

- [ ] **Step 1: 改文档。**
- [ ] **Step 2: 自检**：`grep -n "format_version" docs/*.md` 全部一致。
- [ ] **Step 3: Commit** — `git commit -m "docs: capture v2, raw-depth device checklist"`

---

## 自检记录

- 规格覆盖：§5.3 "用 ARCore 深度初始化"（Task 3）、"位姿优化（开发者实测决定）"（Task 4，实测依据见头部）、拍摄引导"拍得太快提示"（Task 7）。§6 第一期成功标准不受影响（训练时间可能略增，Task 5 记录）。
- 类型一致性：`Frame.confidence`/`depth_kind`（T1）→ `build_training_set`（T2）→ `TrainingSet.confs`（T2/T3/T4/T5）；`DepthBytes.kind/confidence`（T6）→ `addFrame(depthKind, confidence)`（T6）；`Profile.depth_lambda/pose_lr`（T4/T5）。
- Review Focus 五项分别由 T1/T2（1）、T6（2）、T2（3）、T4（4）、T4（5）的测试固定。
- 比例：计划约为改动代码量的同一量级；代码块只有测试断言与签名。
