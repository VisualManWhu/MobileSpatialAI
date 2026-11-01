#!/usr/bin/env bash
# Recon3D 服务端一键安装（可重复执行）。用法：bash server/scripts/install.sh
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SERVER_DIR"

# 虚拟环境放在 Linux 家目录而不是 /mnt/d 下：WSL 访问 Windows 磁盘很慢，
# 把几个 GB 的 torch 放在那里会让安装和启动都慢很多。可用环境变量覆盖。
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/recon3d}"
# 虚拟环境(ext4)与项目缓存/源码(/mnt/d)不在同一文件系统，硬链接会失败并刷屏警告，直接用复制。
export UV_LINK_MODE=copy
# 禁止从源码编译 gsplat（需要 nvcc）；预编译包拿不到时直接报错，而不是悄悄走编译。
export UV_NO_BUILD_PACKAGE=gsplat

fail() { echo; echo "[失败] $*" >&2; exit 1; }

echo "==> 1/4 检查 uv"
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv，正在用官方安装器安装..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv 安装失败，请检查网络后重试。"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv 安装后仍找不到，请重新打开终端再运行本脚本。"
fi
uv --version

echo "==> 2/4 准备 Python 3.10（gsplat 预编译包只支持 3.10）"
uv python install 3.10 || fail "Python 3.10 安装失败，请检查网络。"

echo "==> 3/4 安装依赖（首次约 3 GB，需要几分钟；无需安装 CUDA 工具包）"
uv sync || fail "依赖安装失败。请检查网络后重试（可重复运行本脚本）。
若提示 gsplat 相关错误：gsplat 预编译包下载失败（检查网络 / docs.gsplat.studio 是否可访问），
脚本不会改为从源码编译（那需要 CUDA 工具包 nvcc）。"

echo "==> 4/4 检查 GPU 是否可用"
if ! uv run python -c "import torch, gsplat; assert torch.cuda.is_available()"; then
  cat >&2 <<'MSG'

[失败] 依赖已装好，但 PyTorch 看不到 NVIDIA 显卡。请按下面排查：
  1. 在本终端运行 nvidia-smi。
     - 提示找不到命令或报错：
         * Windows + WSL：请在 Windows 上更新 NVIDIA 显卡驱动
           （https://www.nvidia.cn/drivers/ ，装 Game Ready 或 Studio 驱动均可），
           然后在 PowerShell 运行 wsl --shutdown，再重新打开 Ubuntu。
           不要在 WSL 里安装 Linux 版显卡驱动。
         * 实验室 Ubuntu 服务器：请联系管理员确认 NVIDIA 驱动已安装（需支持 CUDA 12.4 及以上）。
  2. nvidia-smi 正常但仍失败：驱动版本可能过旧，请升级到最新版本。
  详细步骤见 docs/setup/ 下的文档。
MSG
  exit 1
fi

echo
echo "[成功] 安装完成，GPU 可用。下一步：bash server/scripts/run.sh"
