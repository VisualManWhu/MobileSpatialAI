#!/usr/bin/env bash
# 预先下载物体识别用的模型权重（否则第一次点"识别物体"时才下载，要等很久）。
# 用法：bash server/scripts/predownload_models.sh [laptop|server]
#   laptop：Grounding DINO tiny + SAM 2.1 small + Qwen2.5-VL-3B（约 8 GB）
#   server：Grounding DINO base + SAM 2.1 large + Qwen2.5-VL-7B（约 20 GB）
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SERVER_DIR"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/recon3d}"
export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "未找到 uv，请先运行：bash server/scripts/install.sh" >&2; exit 1; }

PROFILE="${1:-laptop}"
case "$PROFILE" in
  laptop|server) ;;
  *) echo "档位只能是 laptop 或 server，收到：$PROFILE" >&2; exit 1 ;;
esac

echo "档位：$PROFILE    权重缓存目录：${HF_HOME:-$HOME/.cache/huggingface}"
exec uv run --no-sync python - "$PROFILE" <<'PY'
import sys

from huggingface_hub import snapshot_download

from recon3d.config import PROFILES

p = PROFILES[sys.argv[1]]
for model_id in (p.objects_detector, p.objects_sam, p.objects_vlm):
    print(f"下载 {model_id} …", flush=True)
    path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges*", "vocab*", "*.py"])
    print(f"  -> {path}", flush=True)
print("[成功] 模型权重已就绪。")
PY
