# Recon3D 重建模式设计：快速 / 高精（COLMAP 特征匹配 + BA 位姿）

日期：2026-11-05。状态：待实现。

## 1. 为什么

现在的位姿全部来自手机 ARCore 的视觉惯性里程计（VIO），服务端照用，不做任何全局优化
（`pose_opt.py` 的光度微调默认关闭，实测有害）。单个房间够用：相邻帧一致性 0.6 px / 0.7°。
但 VIO 是增量累积的，多房间 + 走廊的长轨迹会漂移（米级轨迹下几 cm 到十几 cm、零点几度），
绕一圈回到起点时墙面双影、地板错层、远处漂浮物。用户要拍多房间带走廊的场景，户外次之。

解法：给重建加一个**高精模式**——用 ARCore 位姿和内参做初值，COLMAP 提特征、顺序匹配 +
按 ARCore 位置做空间匹配找回环、以已知位姿三角化、全局 Bundle Adjustment 优化位姿（内参固定），
再对齐回 ARCore 坐标系保住米制尺度和 +Y 向上。其余训练流程不变。

快速模式 = 现状。两种模式吃同一个拍摄包：**拍摄不变**，模式只是上传时的一个选择，一个包可以
先快速看效果、再高精重算，不重传。

## 2. 目标 / 非目标

目标：
- 预览页和房间页"上传"分成「快速上传」「高精上传」；失败后的"重新上传"同样两个。
- 已完成的拍摄可「改用高精重算」/「改用快速重算」：服务端复用已有拍摄包起新任务，不重传。
- 高精流水线：pycolmap（CPU SIFT）→ 顺序匹配 + 空间匹配（ARCore 位置先验）→ 已知位姿三角化
  → 全局 BA（内参固定）→ Sim3 对齐回 ARCore → 位姿替换进现有训练。
- 容错：BA 未注册的帧保留 ARCore 位姿；注册率 < 70% 整体回退快速算法并在任务说明里写明。
- 进度与状态：高精任务前 0–30% 为"位姿优化"，手机和状态页都能看到阶段与模式。
- 离线评估：`eval_scene.py --mode precise` 对已有包比较两种模式。
- 时间：多等 15–30 分钟（400–600 帧）用户接受；190 帧实测见 §8。

非目标：
- 拍摄侧任何改动（关键帧密度、分辩率、引导提示）。
- GPU SIFT / GLOMAP / 词袋回环（空间匹配靠 ARCore 位置先验已能找回环）。
- 高精模式下重算内参、畸变。
- 户外专项（天空、远景）处理：先用同一套，看效果再说。
- 重算后自动删旧任务。

反目标：
- 高精结果尺度或朝向变了，导致房间边界、深度点云、查看器 +Y 向上全乱——必须 Sim3 对齐回 ARCore。
- COLMAP 跑了半小时失败后任务直接"失败"，用户什么都没拿到——必须回退快速算法并说明。
- 快速模式行为有任何变化——所有新逻辑都在 `mode == "precise"` 分支里，默认 `fast`。

## 3. 服务端

### 3.1 任务模式

- 任务新增字段 `mode: "fast" | "precise"`，存在 `status.json`（`JobStatus.mode`，缺省 `"fast"`，
  旧目录没有这个字段按 fast 读）。
- `POST /uploads` 请求体新增可选 `mode`（默认 `fast`）；写进 `upload.json`，finish 时带进 status。
- 旧接口 `POST /jobs`（multipart）新增可选表单字段 `mode`，默认 fast。
- 新接口 `POST /jobs/{id}/rerun`，body `{"mode": "fast" | "precise"}` → `201 {"job_id": <新 id>}`：
  源任务须存在且有 `capture.zip`（任何状态，含 failed）；新任务目录里的 `capture.zip` 是源文件的
  **硬链接**（`os.link`，失败则复制），`log.txt` 第一行 `created from <源 id> (rerun, mode=precise)`。
  源任务不存在 404 `任务不存在`；源任务还在上传中（没有 status.json）409 `上传未完成`。
- `GET /jobs/{id}`、`GET /jobs` 响应增加 `mode`。不加 `stage` 字段：阶段文字就在 `message` 里。
- 状态页加"模式"列（快速 / 高精），说明列本来就显示 `message`，高精阶段自然显示"位姿优化 (…)"。

### 3.2 高精位姿流水线（新模块 `recon3d/recon/precise_pose.py`）

输入：`Capture`（已解包，images/ 目录，`frames[i].pose` GL c2w，内参 fx fy cx cy）、工作目录、进度回调。
输出：`PrecisePoseResult(c2w_cv: [N,4,4], registered: [N] bool, stats: dict)`，c2w 为 OpenCV 约定、
ARCore 世界坐标系（米）。

步骤与参数（pycolmap 4.2.1，`has_cuda == False`，全 CPU）：

1. **特征提取** `extract_features`：相机模型 PINHOLE，`max_image_size = 1600`，`sift.max_num_features = 4096`，
   `num_threads = os.cpu_count()`。相机：ARCore 每帧都给内参，通常整包相同但不保证——按
   (fx, fy, cx, cy) 四舍五入到 0.1 px 分组，每组一台相机（`CameraMode.PER_IMAGE` 后合并，或先建相机再
   `existing_camera_id` 逐组提取）；内参全程固定不优化，训练用的 K 与 BA 用的 K 完全一致。
2. **位置先验**：pycolmap 4.x 先验挂在 frame 上：对每个 frame `write_pose_prior(PosePrior(position=ARCore 位置,
   coordinate_system=CARTESIAN, corr_data_id=frame.data_ids[0], position_covariance=diag(0.3²)))`。
   `frame.image_ids` 给的是 `data_t`，取 `.id` 才是 image_id。
3. **匹配**：`match_sequential(overlap = 10, loop_detection = False)` + `match_spatial(max_num_neighbors = 30,
   max_distance = 3.0 m, ignore_z = False)`。空间匹配就是回环：走廊两头、回到起点的帧在 3 m 内，
   不需要词袋。
4. **初始重建**：建 `Reconstruction`，沿用数据库里的 camera / rig / frame id（`add_camera` 先于 `add_rig`），
   `frame.rig_from_world = inv(ARCore c2w · GL→CV)`，全部 `register_frame`；先 `triangulate_points(refine_intrinsics=False)`
   得到初始点（mapper 对没有点的输入什么都不做，见 §8），写到磁盘作为 mapper 的 `input_path`。
5. **迭代三角化 + 全局 BA**：`incremental_mapping(database, images, out, options, input_path=初始重建)`。
   以已有位姿为起点，COLMAP 的 mapper 做"三角化 → 全局 BA → 重新三角化/合并轨迹 → 过滤 → 再 BA"的迭代
   （`ba_global_max_refinements = 5`），这是回环能闭合的关键：第一次用漂移位姿三角化时，起点与终点之间的
   匹配会因重投影误差大被过滤掉，位姿被局部约束拉近之后，下一轮重新三角化才把这些回环轨迹接上。
   **单次 triangulate + bundle_adjustment 不够**（评审者指出），不用。
   选项：`ba_refine_focal_length = ba_refine_principal_point = ba_refine_extra_params = False`（内参固定）；
   `use_prior_position = True`、`use_robust_loss_on_prior_position = True`、`prior_position_loss_scale = 0.3`
   （米）——ARCore 位置作为弱先验，用 Cauchy 鲁棒核：局部漂移几 cm 时几乎不起作用，米级漂移时它把整体
   坐标系钉在 ARCore 上（尺度、原点、朝向都不会跑），回环由视觉约束闭合。
6. **对齐**：有了位置先验，BA 结果本来就在 ARCore 坐标系里；再做一次**全体注册帧的最小二乘 Umeyama
   （带尺度，不做 RANSAC）**作保险并得到统计量。尺度因子偏离 1 超过 5% 或旋转超过 2° → 日志 WARNING。
   不用 RANSAC 对齐：轨迹长时"最大一致段"会把其他段整体推开。
7. **未注册帧**：mapper 后 `has_pose == False` 或三角化点数 < 20 的帧视为未注册。位姿 = ARCore 位姿套上
   **时间上最近的注册帧的刚体修正量**（`c2w_new[j] · inv(c2w_arcore[j]) · c2w_arcore[i]`，j 为最近注册帧），
   不是原始 ARCore 位姿：否则它会和被修正过的邻帧之间出接缝。`registered[i] = False`。
8. **回退**：`registered.mean() < 0.7` → 抛 `PrecisePoseFailed(ratio)`，上层改用 ARCore 位姿训练，
   任务说明前缀 `高精位姿失败（仅 42% 帧注册），已按快速模式重建；`。pycolmap 任何异常同样回退，
   原始异常写进 `log.txt`。
9. **统计**（写 `log.txt` 一行 + 返回）：注册率、相机中心平均/中位/最大位移（米）、平均旋转修正（度）、
   三角化点数、各阶段耗时、对齐尺度。这是用户看漂移有多大的依据。

进度：提取 0→0.35，匹配 0.35→0.7，mapper 0.7→1.0（相对高精阶段内部；mapper 无回调，按
`max_runtime_seconds` 估算推进到 0.95，结束时 1.0）。

### 3.3 流水线接入

`jobs/pipeline.py`：

```
if mode == "precise":
    on_stage("位姿优化")                      # progress 0 → POSE_SHARE (0.3)
    result = precise_pose(cap, workdir, lambda p: on_progress(POSE_SHARE * p))
    ts = build_training_set(cap, profile, c2w_override=result.c2w_cv)
    note = "" or 回退说明
else:
    ts = build_training_set(cap, profile)
train: on_progress(POSE_SHARE + (TRAIN_SHARE - POSE_SHARE) * p)   # 0.3 → 0.9
```

- `build_training_set` 加可选参数 `c2w_override: np.ndarray | None`（[N,4,4] OpenCV c2w）；
  给了就替代 `gl_to_cv_c2w(fr.pose)`。**深度位姿**（`depth_c2w`，raw depth 滞后那帧的位姿）
  相应修正：对第 i 帧，`depth_c2w_new = c2w_new[i] · inv(c2w_arcore[i]) · depth_c2w_arcore`
  （把同一帧的刚体修正量套到深度位姿上）。
- Worker 状态：高精阶段 `state = "training"`、`message = "位姿优化 (特征提取)"` 等三个阶段文字，
  进度 0→0.3；之后 `"训练中"` 照旧。手机不需要认识新状态。`JobStatus` 增加 `stage: str | None`
  仅给状态页用（message 已经含阶段，stage 可省——**决定：不加 stage 字段**，message 足够）。
- 回退说明：训练完成后 message 为 `完成` 不变；回退时 `完成（高精位姿失败，已按快速模式重建）`，
  `log.txt` 记详细原因。

### 3.4 评估

`scripts/eval_scene.py --mode fast|precise`：高精时先跑 `precise_pose`，打印统计，再按现有流程重训/评估。
新脚本 `scripts/pose_drift.py <job_dir>`：只跑位姿流水线，输出每帧位移/旋转修正 CSV + 一张
轨迹俯视图 PNG（ARCore 轨迹 vs BA 轨迹），验收多房间包时看漂移形状。

## 4. 手机端

### 4.1 UI

- 预览页：「上传」拆成「快速上传」「高精上传」两个按钮（高精在右、主色），「重拍」保留。
- 房间页 `CaptureRow`：`local` 状态两个按钮「快速上传」「高精上传」；`failed` 状态按钮为
  「重试」（服务端任务还在时，模式沿用记录里的）或「快速重传」「高精重传」（需重传时）。
- `done` 状态：⋮ 菜单新增「改用高精重算」（记录是 fast）或「改用快速重算」（记录是 precise）；
  点击直接开始，不弹确认（可随时彻底删除）。
- 状态文案：`captureStatusText` 在 uploading/queued/training/exporting/done 后追加 ` · 高精`
  （仅 precise；fast 不加字，保持现状）。高精阶段服务端 message 为 `位姿优化 (…)`，training 状态下
  若 message 以 `位姿优化` 开头则显示它而不是 `训练中 x%`。

### 4.2 数据与逻辑

- `CaptureRecord.mode: String = "fast"`（旧记录缺省 fast）。
- `UploadViewModel.start(roomId, captureId, mode)`、`retry` 沿用记录里的 mode；`rerun(roomId, captureId, mode)`：
  要求 `jobId != null`，调 `Api.rerunJob(jobId, mode)` → 新 jobId，记录 `mode = mode, jobId = new,
  serverIds += new, state = queued, progress = 0, scenePath 保留到新模型下载完再替换`（下载完成时
  旧 `scene.splat` 被覆盖写入同路径，期间"查看 3D"仍能看旧模型——下载写 `.part` 再 rename，现有逻辑）。
  rerun 期间 `done` 行变 `queued/training · 高精`，"查看 3D" 消失（现有 state 判断），可接受。
- `Api.upload(zip, uploadId, mode, listener)`：`POST /uploads` 带 `mode`。`Api.rerunJob(jobId, mode): String`。
- 彻底删除：`serverIds` 已经累积所有任务 id，重算出来的旧任务会一起删。

### 4.3 文案

快速上传 / 高精上传 / 快速重传 / 高精重传 / 改用高精重算 / 改用快速重算 / ` · 高精` /
位姿优化 (特征提取) / 位姿优化 (特征匹配) / 位姿优化 (全局优化) /
完成（高精位姿失败，已按快速模式重建）。

## 5. 测试

服务端：
- `test_precise_pose.py`（无 GPU，CPU pycolmap）：合成场景——纹理丰富的立方体房间内一条**闭环轨迹**
  （约 40 帧，绕一圈回到起点），真值位姿渲染图像（简单的纹理贴图投影即可，不需要 3DGS），输入位姿 = 真值 +
  **沿轨迹累积的漂移**（终点偏 0.3 m / 3°，模拟 VIO）→ `precise_pose` 注册率 ≥ 90%、首尾帧相对位姿误差
  比输入小 5 倍以上（回环闭合）、全体相机中心误差中位数 < 输入的 1/3、尺度因子 ∈ [0.98, 1.02]、
  人为抹掉两帧的特征（纯色图）→ 这两帧未注册且位姿 = 邻帧修正量套用后的结果。标记 `@pytest.mark.slow`
  （约 1–2 分钟）。
- `test_precise_pose_fallback`：注册率阈值触发 `PrecisePoseFailed`；pycolmap 抛异常时同样回退。
- `test_dataset.py`：`c2w_override` 替换位姿且 `depth_c2w` 按刚体修正量跟着变。
- `test_jobs.py` / `test_api.py`：`mode` 存取（默认 fast、旧 status.json 无字段）、`POST /uploads {mode}`、
  `POST /jobs/{id}/rerun` 成功（硬链接或复制、新 id、log 首行）、源不存在 404、上传中 409、
  `GET /jobs` 带 mode、worker 用 fast 流水线跑 precise 任务的进度分段（pipeline 注入 fake）。

手机端：
- `ApiClientTest`：upload 的 `POST /uploads` body 含 `mode`；`rerunJob` 请求路径/体与响应解析、404/409。
- `UploadViewModelTest`：`start(mode)` 记录 mode 并传给 Api；`rerun` 换 jobId、追加 serverIds、状态 queued、
  旧 scenePath 保留直到新模型下载完；`captureStatusText` 的 ` · 高精` 与 `位姿优化` 显示；旧记录缺省 fast。
- `RoomRepositoryTest`：mode 持久化。

## 6. 文档

- `docs/api.md`：`mode` 字段、`POST /jobs/{id}/rerun`、状态页模式列。
- `docs/setup/01-server-laptop.md`：pycolmap 随 `uv sync` 安装（加进 `pyproject.toml` dependencies）。
- `docs/acceptance/phase1.md` §4 新增"高精模式"验收项；§5 记录 190 帧包的耗时与位移统计（§8）。

## 7. 我拍板的小决定

- 用 ARCore 位置做空间匹配找回环，不下载词袋树：少一个 100 MB 外部文件，且位置先验比词袋更准。
- 内参固定不优化：ARCore 内参标定好，自由度越少 BA 越稳；按内参分组建相机，训练与 BA 用同一套 K。
- 用 COLMAP mapper 的迭代"三角化 → BA → 重三角化"而不是单次 triangulate + BA（评审者指出单次闭不了环）。
- ARCore 位置作鲁棒弱先验钉住坐标系，对齐改为全体最小二乘（不 RANSAC）；未注册帧套邻帧修正量避免接缝。
- 对齐回 ARCore 而不是让 ARCore 向 BA 对齐：房间边界、深度点云、查看器 +Y 都基于 ARCore 坐标。
- 不加 `stage` 字段（§3.1/§3.3 原先一处矛盾，已统一）。
- 回退阈值 70%：低于它说明匹配整体失败，部分修正反而会让注册/未注册帧之间出接缝。
- 高精阶段占进度 30%：190 帧实测比例见 §8，600 帧时 COLMAP 占比更高，进度条前段慢属正常。
- 重算不弹确认：动作可逆（旧模型在服务端，还能再算；彻底删除清理）。
- 不加 `stage` 字段：message 已含阶段文字。
- 硬链接复用拍摄包：200 MB 不复制；跨设备失败就复制。

## 8. 190 帧包实测（spike，2026-11-05，任务 35aa74d5a8d5，1920×1080，单房间）

| 阶段 | 耗时 | 备注 |
|---|---|---|
| 特征提取（CPU SIFT，max_image_size 1600，4096 特征，8 线程） | 60 s | 每帧 0.3 s |
| 顺序匹配（overlap 10） | 33–75 s | 两次运行差异大，机器负载影响 |
| 空间匹配 | 未测到 | spike 里先验写晚了；正式实现先写先验再匹配 |
| 已知位姿三角化 `triangulate_points` | 13 s | 10.5 万点，重投影 1.35 px |
| mapper 迭代 BA（`incremental_mapping`，input = 三角化结果，位置先验） | 14 s | 7.7 万点，重投影 0.61 px |
| **合计** | **约 2.5–3 分钟** | 600 帧估 10–15 分钟 |

位姿修正量（相对 ARCore，Sim3 对齐后）：均值 1.3 cm，中位 1.2 cm，最大 3.5 cm；尺度 1.0000，旋转 0.00°。
单房间 190 帧 ARCore 已经很准，高精只是 1 cm 级修正——符合预期；多房间走廊才是它的用处，待验收时测。

两个教训写进设计：
- **不带先验的 `bundle_adjustment` 尺度漂了 23%**（1.235），旋转 0.49°：BA 的规范自由度。有位置先验的 mapper 尺度 1.0000。
  所以先验不是可选项；对齐只是保险。
- mapper 以"没有点的初始重建"作 input 时什么都不做（0.1 s、0 点）：必须先 `triangulate_points` 再喂 mapper。
  pycolmap 还提示 `No pose priors with valid covariance found`（终局对齐被跳过）：写先验时带
  `position_covariance = diag(0.3²)`。

## 9. 1123 帧多房间包实测（spike，2026-11-07，任务 40a80bd6c17f，两房间 + 20 m 走廊，轨迹 127 m）

同一拍摄、同一训练预算（laptop profile，15000 步），只换位姿：

| | 快速（ARCore 位姿） | 高精（COLMAP BA 位姿） |
|---|---|---|
| PSNR 中位数 | 16.13 dB | **22.66 dB** |
| PSNR 最小 | 8.17 dB | 9.82 dB |
| 训练后高斯数 | 10.3 万（裁掉 253 万） | **143 万** |
| 不透明高斯 | 2 471 | 43 307 |
| floater 分数 | 0.70 | **0.22** |
| 训练用时 | 13.3 min | 23.5 min（和 COLMAP 匹配抢 CPU） |

快速模式在这个场景**整体坍塌**：位姿漂移让深度点云和图像互相矛盾，支持裁剪几乎把所有高斯都当成飘点删掉，
偏轨渲染全是色块。高精位姿下同一流程出了能看的房间、走廊、家具（`data/ab/precise/40a80bd6c17f/compare_offtrack.jpg`）。

COLMAP 阶段耗时（CPU，与 GPU 训练并行、机器中途休眠，数字偏大）：

| 阶段 | 耗时 | 备注 |
|---|---|---|
| 特征提取 | 453 s | 每帧 0.4 s |
| 顺序匹配（overlap 10） | 546 s | |
| 空间匹配（30 邻居，3 m） | 实测无效（机器休眠） | 恢复后约 0.7 s/帧 → 估 13 min |
| 已知位姿三角化 | 101 s | 27.7 万点，重投影 1.30 px |
| mapper 迭代 BA（位置先验） | 881 s | 26.2 万点，1123/1123 注册，重投影 0.68 px |
| **合计（估）** | **约 45 min** | 超出 §1 的 15–30 min 预算，见下 |

位姿修正量（相对 ARCore）：均值 8.9 cm，中位 6.8 cm，最大 50 cm（第 682 帧，走廊尽头房间）；
尺度 1.0000，旋转 0.00°（先验钉住坐标系）。高度漂移范围 0.60 m → 0.43 m。
走廊去/回两段的最近距离中位数 0.28 m → 0.23 m：ARCore 的回环误差在这个包里不算大，
主要收益来自**局部**几厘米到几十厘米的位姿修正——对 3DGS 来说这就是坍塌与否的差别。

由此调整的决定：
- 1123 帧 45 min 超预算。正式实现里：空间匹配 `max_num_neighbors` 降到 20；mapper
  `ba_global_max_refinements` 3 次足够（本次 0.68 px 已收敛）；顺序匹配 overlap 保持 10。
  预计 25–30 min。若仍超，下一步是在高精模式里按 ≥ 0.15 m / 10° 抽稀再做 COLMAP（未注册帧套邻帧修正量）。
- 匹配/BA 运行时 GPU 空闲：worker 可以在 COLMAP 阶段先跑单目深度推理（52 s），省不了多少，不做。
- pycolmap 4.2.1 API 注意：`Image.cam_from_world` 是**方法**（`cam_from_world()` 返回 `Rigid3d`，再 `.matrix()` 取 3×4）；
  `Reconstruction.add_camera` 必须在 `add_rig` 之前；`FeatureExtractionOptions.max_image_size`（不在 `sift` 下）。
