# Recon3D

手机 ARCore 拍摄房间，服务端用 3D Gaussian Splatting 重建三维场景。

- `server/`: Python 服务端（拍摄包解析、重建流水线、API）
- 手机端 3D 查看器（WebView + Spark，含重新打包步骤）: [docs/viewer.md](docs/viewer.md)
- 拍摄包格式: [docs/capture-format.md](docs/capture-format.md)
- 服务端 API: [docs/api.md](docs/api.md)
- 设计文档: docs/superpowers/specs/

服务端在 WSL Ubuntu 中运行与测试：

```
cd server && UV_PROJECT_ENVIRONMENT=$HOME/.venvs/recon3d uv run pytest
```
