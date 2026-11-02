# 第一期验收

依据：[设计文档](../superpowers/specs/2026-11-01-recon3d-design.md) 第 6 节"第一期"和第 7 节。
自动化测试日期：2026-11-01（分支 `phase1`）；手机实测 **待测**（暂时没有手机可用）。

## 1. 验收标准与测量方法

设计文档第 6 节第一期：一个约 15 m² 的卧室，按引导 3 分钟内拍完；在笔记本上 30 分钟内重建完成；在手机上浏览流畅，家具和桌上的物品都认得出来。

| # | 标准 | 测量方法 | 实测结果 | 结论 |
|---|------|----------|----------|------|
| A1 | 约 15 m² 卧室，3 分钟内拍完 | 用手机秒表：从点红色录制按钮开始，到点"完成拍摄"出现"已保存拍摄，可以上传了"Toast 为止。另记已拍张数；截图保存拍摄界面和房间列表 | 待测 | 待测 |
| A2 | 笔记本上 30 分钟内重建完成 | 笔记本档（15000 次迭代）。看 App 里任务行：从"排队中"出现到"完成"的时间；同时对照服务端窗口日志的起止时间。截图任务行（含训练百分比）和服务端日志 | 9 分 06 秒（2026-11-02，193 帧，服务端日志 15:08:58 → 15:18:04） | 通过 |
| A3 | 手机上浏览流畅 | 在查看器里单指旋转、双指缩放和平移，主观感觉是否跟手无明显卡顿；另可通过 chrome://inspect 看帧率。录屏 10 秒以上 | 用户主观：流畅（2026-11-02，office 场景，142 万高斯） | 通过 |
| A4 | 家具和桌上物品认得出来 | 在查看器中走近床、桌、椅和桌面物品（杯子、书等）并截图，逐项记"认得出 / 模糊 / 认不出" | 用户主观：认得出（2026-11-02，窗户、百叶窗、桌面；未逐项截图） | 通过（初步） |
| A5 | 第 7 节：出错时提示用户看得懂 | 做第 4 节第 6 项的断网、关服务端实验，记录 App 实际显示的文字 | 待测 | 待测 |

哪一项不达标，记成问题放在第 5 节，不在本阶段修。

## 2. 自动化测试结果（2026-11-01，HEAD `37d690b`）

| 套件 | 命令 | 结果 |
|------|------|------|
| 服务端（WSL，含 GPU 测试） | `cd server && UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d uv run pytest -q -rs` | 70 passed in 78.30s（最终修复轮后） |
| Android 单元测试 | `.\gradlew.bat :app:testDebugUnitTest` | BUILD SUCCESSFUL；113 个测试，0 失败（最终修复轮后） |
| Android 调试包 | `.\gradlew.bat :app:assembleDebug` | BUILD SUCCESSFUL；`android/app/build/outputs/apk/debug/app-debug.apk`（约 35 MB） |
| 查看器网页 | `cd android/viewer-web && npm test` | 23 passed，0 failed |

若服务端测试输出里看到 `skipped`，说明 CUDA 不可用、GPU 测试被跳过，**不算通过**。

### 合成数据 GPU 训练耗时

- 实测（任务 4，合成数据，RTX 4060 Laptop）：2000 次迭代约 22 秒。
- **估算**（未实测）：笔记本档 15000 次迭代，按线性外推约 3 分钟。真实房间的图片数量、分辨率和高斯数量都更大，实际会明显更长，此数字只能当下限参考。
- 真实房间重建耗时：**待测**（见 A2）。

## 3. 已知限制

- 以上自动化测试覆盖服务端流水线（合成数据）、App 的 JVM 逻辑和查看器的纯函数；ARCore 拍摄、相机方向、真实上传、WebView 渲染都**只能在真机上验证**。
- 查看器坐标系：判断为 ARCore 世界系（+Y 向上）与 three.js 一致，不做翻转。若真机上地板在上面，说明这个判断有误，记为问题，不要临时加翻转。

## 4. 手机测试清单

按顺序做。每项括号内是期望结果。出错时先看对应的安装文档的"打不开怎么办"。

### 1) Windows 管理员步骤（你本人在管理员 PowerShell 执行）

详细步骤和命令见 [docs/setup/01-server-laptop.md](../setup/01-server-laptop.md)，这里只列要点：

- [x] `.wslconfig` 不用镜像模式（`networkingMode=NAT`）——2026-11-02 实测镜像模式时通时不通并劫持 8765，已切换
- [x] 端口转发 `netsh interface portproxy add v4tov4 listenport=8765 ... connectport=18765 connectaddress=127.0.0.1`
- [x] 网络二选一：Wi-Fi 标记为"专用网络"，或使用 Tailscale
- [x] 防火墙放行 8765：Wi-Fi 方式用 `New-NetFirewallRule ... -Profile Private`；Tailscale 方式用 `-Profile Any -RemoteAddress 100.64.0.0/10`（只允许 Tailscale 设备）
- 实测：手机浏览器与 App 均连通（`192.168.178.16:8765`，已连接：RTX 4060 Laptop GPU（笔记本档））
- 撤销：`netsh interface portproxy delete v4tov4 listenport=8765 listenaddress=0.0.0.0`；`Remove-NetFirewallRule -DisplayName "Recon3D server 8765"`（Tailscale 规则名为 `"Recon3D server 8765 Tailscale"`）

### 2) 手机浏览器检查服务端

- [x] 在 Ubuntu 运行 `bash server/scripts/run.sh`，窗口保持打开
- [x] 手机浏览器打开 `http://<IP>:8765/api/v1/health`
- 期望：返回 JSON，其中 `gpu` 含 "RTX 4060" —— 实测通过（2026-11-02）

### 3) 手机装 Tailscale（选了 Tailscale 才做）

- [ ] 手机装 Tailscale，用与笔记本**同一个账号**登录
- 期望：App 里要填的地址是 `100.x.x.x:8765`（笔记本上 `tailscale ip -4` 查）

### 4) 安装 App 并连接

- [x] 按 [docs/setup/03-phone.md](../setup/03-phone.md) 安装 APK（实测用 `adb install`，Pixel 10 Pro / Android 16）
- [x] 设置页填地址（不写端口也行，会补成 `:8765`），点"测试连接"
- 期望：显示"已连接：RTX 4060 Laptop GPU（笔记本档）"（App 只去掉 GPU 名称里的 NVIDIA 和 GeForce）—— 实测通过
- [ ] 地址为空时点上传：弹出"请先在设置中填写计算端地址"并跳到设置页

### 5) 拍摄界面

先新建房间（名字随意），进入房间点"拍摄"，授予相机权限。

- [ ] 实时画面方向正确（竖屏，不拉伸、不镜像）
- [ ] 原地转一整圈：圆环从 0 走到 100%，界面进入"② 上下"（"转圈"标签变成"✓ 转圈"）
- [ ] 手机快速晃动：出现黄色提示"太快了，慢一点"
- [ ] 手掌盖住镜头：出现"跟丢了…"，箭头隐藏
- [ ] 在支持深度的手机上，"正在等待深度传感器…"提示在跟踪稳定后几秒内消失，之后才能开始录制
- [ ] 点红色按钮，状态显示"录制中 · 已拍 N 张"；走完各阶段后点"完成拍摄"
- 期望：Toast "已保存拍摄，可以上传了"，房间列表里该次拍摄显示"本地"和"上传"按钮
- [ ] 取出拍摄包：`adb exec-out run-as dev.recon3d cat files/captures/<房间>/<拍摄ID>.zip > cap.zip`（文件在应用私有目录，普通 `adb pull` 读不到）
- [ ] 把上一步取出的 `cap.zip` 复制到 `D:\cap.zip`。然后从开始菜单打开 "Ubuntu" 终端（不是 PowerShell），粘贴下面这一整行检查拍摄包：

  ```bash
  cd /mnt/d/projects/Recon3D/server && UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d uv run python -c "from pathlib import Path; from recon3d.capture.format import load_capture; c = load_capture(Path('/mnt/d/cap.zip'), Path('/tmp/capcheck')); print('帧数', len(c.frames), '图像', c.manifest.image_width, 'x', c.manifest.image_height)"
  ```

  期望：不报错，输出一行"帧数 N 图像 W x H"。仓库自带的样例包（`server/tests/fixtures/android_v1.zip`）实测输出为 `帧数 3 图像 64 x 48`，说明命令本身可用。
  真机拍摄包的期望：帧数与 App 显示的"已拍 N 张"一致，图像长边（W、H 中较大者）>= 1280；对着远处墙时深度 PNG 里出现 > 8191 mm 的值。
- [ ] logcat 中 `CaptureScreen: AR ready: image WxH, depth=…` 记下实际分辨率；`CapturePipeline` 的丢弃/失败计数也记下
- [ ] 拍摄结束后 logcat 中 `CaptureRenderer: raw depth: fresh N / stale M / duplicate D; lag ms …`（v2 拍摄包的原始深度计数；fresh = raw depth 按其自身时间戳在 2 s 位姿缓冲里配到了位姿并写入，duplicate = 同一张 raw 图已在上一关键帧写过、本帧回退平滑深度，stale = 超过 2 s 或不可用）：
  期望 fresh 占多数（fresh > stale + duplicate；2026-11-02 改配对方式前实测为 `fresh 0 / stale 340`，因为 raw depth 永远落后当前帧），低于则记在第 5 节。
  拍摄包 zip 里 `conf/` 文件数应等于 fresh 数；用上面的 Ubuntu 命令加 `print(sum(f.depth_pose is not None for f in c.frames))` 可数出带 `depth_pose` 的帧数，应与 fresh 一致
- [ ] logcat 中 `CaptureRenderer: intrinsics: …`（每次拍摄一行）记下：CPU 图像尺寸、CPU 内参（imageIntrinsics 的尺寸与 f、c）、纹理内参（textureIntrinsics 的尺寸与 f、c）、深度图尺寸与写入的 depth k。
  若纹理与 CPU 图像宽高比不同，说明深度内参确实需要单独记录（拍摄包里每帧的 `depth_fx/fy/cx/cy`）
- [ ] 故意拍一次没有深度的：在很暗的房间里拍，或用手指挡住机背的深度/ToF 传感器（若有）。等 10 秒后若仍无深度，录制时应出现黄色提示"未获取到深度，本次拍摄将不含深度（重建质量可能下降）"，logcat 有 `depth never became available` 警告。
  上传重建后判断结果是否还能用（记"可用 / 勉强 / 不可用"）

### 6) 上传与重建流程

- [ ] 点"上传"，依次看到：上传中 x%（带进度条）、排队中、训练中 x%、导出中 / 正在下载模型…、完成
- 期望：出现"查看 3D"按钮；`files/rooms/<房间ID>/<拍摄ID>/scene.splat` 已存在
- [ ] 上传中途开飞行模式：显示"失败：…"（原因文字以 App 实际为准），拍摄包保留，点"重新上传"能重传
- [ ] 训练中短暂断网（开飞行模式约 20 秒再关）：状态显示"网络暂时不可用，正在重试 (n/10)"，恢复网络后继续显示训练进度并最终完成，不需要重新上传
- [ ] 上传中途关掉服务端：显示"失败：连不上计算端，请检查 Wi-Fi 和地址"，点"重新上传"能重传
- [ ] 训练中关服务端再重启：显示"失败：服务重启，任务中断，请重新上传"，点"重新上传"能重传；训练中杀掉 App 再打开，轮询应自动恢复

### 7) 查看器

- [ ] 点"查看 3D"：先出现"正在加载场景…"，随后消失，场景渲染出来
- [ ] 方向正确：地板在下、天花板在上（务必确认；若地板在上，记为问题，见第 3 节）
- [ ] 初始视角：在拍摄起点附近、朝向 -Z 平视，转动时绕前方某点旋转，看起来合理（记下是否需要调整）
- [ ] 单指旋转、双指缩放和平移顺滑，阻尼自然，无明显卡顿
- [ ] 用调试包连电脑，Chrome 打开 `chrome://inspect`，在控制台运行 `flyTo({x:0,y:1.5,z:2},{x:0,y:1,z:0},1500)`：镜头平滑缓动飞过去；`setCameraHome()` 回到初始视角
- [ ] Network 里只有 `appassets.androidplatform.net` 的请求；控制台 fetch `/scene/../rooms.json` 或 `/scene/x` 返回 404
- [ ] 错误路径：把 `scene.splat` 改名或截断，打开查看器，几秒内显示"场景文件加载失败"，点"重试"可重新加载（没有场景时显示"这次拍摄还没有可查看的场景"）
- [ ] 手机静置在查看器时 CPU/GPU 空闲（按需渲染）；返回上一屏不崩溃

### 8) 验收测量（对应第 1 节）

用一个约 15 m² 的卧室完整走一遍，把实测数字和截图填回第 1 节的表格：

- [ ] A1 拍摄用时（秒）、张数
- [ ] A2 重建用时（分钟），对照服务端日志
- [ ] A3 浏览流畅度，录屏
- [ ] A4 家具和桌面物品识别情况，截图
- [ ] A5 出错提示文字

## 5. 发现的问题

（实测后填写。每项记：现象、复现步骤、对应的验收项。）

2026-11-02 第一次真机（Pixel 10 Pro，Android 16，WebView 156）：

1. **查看器黑屏（已修）**：点"查看 3D"后"正在加载场景…"消失，画面全黑。排查：同一 `scene.splat` 在笔记本浏览器和手机 Chrome 里都正常；
   WebView 里 WebGL 正常、帧循环在跑、`readPixels` 读到的像素是场景颜色，但画布（包括普通 2D 画布）没有合成到屏幕，DOM 元素却能显示。
   改 `canvas { position: fixed; inset: 0 }` 后正常。修复提交见 git。对应第 4 节第 7 项。

#### floater 对比（2026-11-02）

同一 office 拍摄包（193 帧，v1 格式 → 启发式置信度），笔记本档（15000 步，上限 150 万高斯），用 `server/scripts/eval_scene.py` 评估。
PSNR 是在拍摄位姿下渲染全部 193 个训练视角求得（960×540）；floater 分数 = 不透明度 > 0.5 的高斯中，离"置信度 ≥ 0.5 的初始点云"（793 838 点）最近距离 > 10 cm 的比例，越低越好。

| 变体 | PSNR 中位数 | PSNR 最小 | floater 分数 | 不透明高斯 | 高斯总数 | 训练用时 |
|---|---|---|---|---|---|---|
| 基线：本分支之前训练的 `scene.ply` | 22.39 dB | 16.29 dB | 0.1415 | 839 385 | 1 500 000 | 9.1 分钟（任务日志） |
| new：深度损失 0.05 + 位姿修正 1e-4（`--pose-lr 1e-4`，已不再是默认） | 20.19 dB | 15.10 dB | 0.1181 | 593 338 | 1 500 000 | 13.6 分钟 |
| nopose：深度损失 0.05，位姿修正关闭（`--pose-lr 0`） | 22.28 dB | 16.42 dB | 0.0923 | 830 565 | 1 500 000 | 9.1 分钟 |

new 的位姿统计：`位姿修正：旋转 中位数 0.358° / 最大 2.602°，平移 中位数 0.91 cm / 最大 5.97 cm`。

- nopose：floater 分数降 35%，PSNR 中位数只降 0.11 dB，用时不变。
- new：PSNR 中位数降 2.2 dB（超过 0.5 dB 的门槛）。注意修正后的位姿没有导出，评估只能用拍摄位姿渲染，模型若适配了修正位姿，这个数会偏低；但 floater 分数也不如 nopose，训练还慢 50%。
- #189、#190、#55 在三个变体里都在最差之列（new 最差 5 帧为 #190、#55、#189、#53、#54），更像是拍摄本身的问题，与训练设置关系不大。
- 注意：floater 分数的参考点云是新的 ≤ 4 m、按置信度过滤的初始点云；基线是用旧的 ≤ 8 m 点云初始化的，它在 4–8 m 的正常几何也会被算成 floater，所以"基线 → nopose"的降幅偏大，公平的比较是 new 对 nopose。
- **决定**：默认 `pose_lr = 0`（位姿修正关闭，需要时显式开启），深度损失保持开启（`depth_lambda = 0.05`）。
- 复现：`cd server && uv run python scripts/eval_scene.py data/jobs/<任务ID> [--train --tag <名> --pose-lr 0]`，结果写到 `eval.json`。

2. **上传在 Wi-Fi 上极慢并反复中断**：手机↔笔记本经家用路由器（手机 2.4 GHz −69 dBm，笔记本 5 GHz 43%）吞吐不到 40 KB/s，
   71 MB 拍摄包十次都在几 KB～2 MB 处断（`Software caused connection abort` / 服务端 `ClientDisconnect`）；笔记本本机经同一 8765 路径 66 MB 只需 2.6 s。
   临时用 USB：`adb reverse tcp:8765 tcp:8765`，App 地址填 `127.0.0.1:8765`，上传秒完。待解：换 5 GHz / 靠近路由器 / Tailscale 再测。对应第 4 节第 6 项。
3. **锁屏或切到后台会中断上传**：用户实测。现版本要求前台 + 常亮。改进：上传/轮询页保持屏幕常亮（小改）；根治需前台服务（已在独立分支上做，未合并到本分支）。对应 A5 / 第 4 节第 6 项。
4. **中断的上传在 `/mnt/d` 上留下空任务目录**：`Upload.abort()` 的 `rmtree(ignore_errors=True)` 在 9p 文件系统上静默失败，`data/jobs/<id>/` 只剩半个 `capture.zip`。影响：占空间；worker 不会处理它们。待修。
5. **WSL 镜像网络不可靠**：见第 4 节第 1 项，已改为 NAT + portproxy。
6. **笔记本 DHCP 地址会变**（.10 → .16）：文档已加固定 IP 建议。

实测数据（office，193 帧，1920×1080 + 160×90 深度，71 MB）：上传完成 → 完成 **9 分 06 秒**（笔记本档，GPU 85–95%，显存 ≤1.2 GB），
导出 1 418 201 个高斯，`scene.splat` 45 MB。内参：f≈1378 px，深度内参恰为 1/12，无需单独的 depth_fx/fy。
方向正确（地板在下），起始视角在拍摄起点朝 −Z。A1/A3/A4 的主观项待用户填写。
