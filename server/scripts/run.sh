#!/usr/bin/env bash
# 启动 Recon3D 服务端。用法：bash server/scripts/run.sh [laptop|server] [额外参数...]
# 例：bash server/scripts/run.sh server --port 9000
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SERVER_DIR"

# 与 install.sh 相同：虚拟环境放在家目录（避免 /mnt/d 慢），可用环境变量覆盖。
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/recon3d}"
export UV_LINK_MODE=copy  # venv 在 ext4、项目在 /mnt/d，硬链接不可用，避免警告
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "未找到 uv，请先运行：bash server/scripts/install.sh" >&2; exit 1; }

PROFILE=laptop
if [[ $# -gt 0 && "$1" != -* ]]; then
  PROFILE="$1"; shift
fi
case "$PROFILE" in
  laptop|server) ;;
  *) echo "档位只能是 laptop 或 server，收到：$PROFILE" >&2; exit 1 ;;
esac

# WSL 下默认监听 18765，由 Windows 把 8765 转发过来（见 docs/setup/01-server-laptop.md 第 2 步）；
# 普通 Linux 服务器直接监听 8765。显式传 --port 时以参数为准。
IN_WSL=0
grep -qi microsoft /proc/version 2>/dev/null && IN_WSL=1
PORT=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  [[ "${args[i]}" == "--port" ]] && PORT="${args[i+1]:-}"
  [[ "${args[i]}" == --port=* ]] && PORT="${args[i]#--port=}"
done
if [[ -z "$PORT" ]]; then
  if [[ $IN_WSL == 1 ]]; then PORT=18765; else PORT=8765; fi
  set -- "$@" --port "$PORT"
fi

echo "档位：$PROFILE    数据目录：$SERVER_DIR/data    停止：Ctrl+C"
if [[ $IN_WSL == 1 ]]; then
  echo "WSL 模式：服务监听 $PORT，手机访问的是 Windows 的 8765（需已按文档第 2 步设置端口转发和防火墙）。"
  echo "手机浏览器打开：http://<笔记本的 Wi-Fi IP>:8765/api/v1/health   （IP 在 PowerShell 里 ipconfig 查看）"
  echo "本机自检：http://127.0.0.1:$PORT/api/v1/health → http://127.0.0.1:8765/api/v1/health → http://<笔记本IP>:8765/api/v1/health"
else
  echo "手机（须与本机在同一 Wi-Fi）浏览器可打开下面任一地址做连通性测试："
  for ip in $(hostname -I 2>/dev/null); do
    [[ "$ip" == *:* ]] && continue
    echo "    http://$ip:$PORT/api/v1/health"
  done
fi
echo

exec uv run recon3d-server --profile "$PROFILE" --data-dir "$SERVER_DIR/data" --host 0.0.0.0 "$@"
