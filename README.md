# Recon3D

手机用 ARCore 拍摄房间（关键帧、位姿、深度），上传到电脑或服务器，服务端用 3D Gaussian Splatting（gsplat）重建出三维场景，再传回手机，在 App 里用手指转着看。
第一期目标：拍一个房间，几十分钟后在手机上浏览它的 3D 场景。第二期（认物体和问答）、第三期（重访和变化检测）见设计文档。

## 当前状态

第一期代码已全部完成（服务端、Android App、3D 查看器），自动化测试全绿。
**手机实测还没做**，验收结果见 [docs/acceptance/phase1.md](docs/acceptance/phase1.md)（其中"手机测试清单"是实测的执行顺序）。

## 仓库结构

- `server/`: Python 服务端（拍摄包解析、gsplat 重建流水线、HTTP API、安装与启动脚本）
- `android/`: Android App（Kotlin + Compose；ARCore 拍摄、房间列表、上传与进度、查看器）
  - `android/app/`: App 本体
  - `android/viewer-web/`: 手机端 3D 查看器网页（Spark + three.js），构建产物打包进 App
- `docs/`: 文档
  - `docs/setup/`: 安装文档
  - `docs/acceptance/`: 验收文档
  - `docs/superpowers/specs/`: 设计文档

## 快速开始

按顺序看：

1. [docs/setup/01-server-laptop.md](docs/setup/01-server-laptop.md): 在 Windows 笔记本（WSL2 + NVIDIA 显卡）上装好并启动服务端，含防火墙与 Tailscale
2. [docs/setup/02-server-lab.md](docs/setup/02-server-lab.md): 在 Ubuntu 服务器上部署
3. [docs/setup/03-phone.md](docs/setup/03-phone.md): 在 Pixel 上开启开发者模式并安装 App

## 运行测试

服务端（在 WSL Ubuntu 中，含 GPU 测试，约 80 秒）：

```
cd server && UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d uv run pytest -q
```

Android 单元测试与调试包（PowerShell）：

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
cd android
.\gradlew.bat :app:testDebugUnitTest
.\gradlew.bat :app:assembleDebug
```

查看器网页：

```
cd android/viewer-web && npm test
```

## 文档

- [docs/acceptance/phase1.md](docs/acceptance/phase1.md): 第一期验收（标准、测量方法、自动化结果、手机测试清单）
- [docs/viewer.md](docs/viewer.md): 手机端 3D 查看器（含重新打包步骤）
- [docs/capture-format.md](docs/capture-format.md): 拍摄包格式
- [docs/api.md](docs/api.md): 服务端 API
- 设计文档: [docs/superpowers/specs/2026-11-01-recon3d-design.md](docs/superpowers/specs/2026-11-01-recon3d-design.md)
