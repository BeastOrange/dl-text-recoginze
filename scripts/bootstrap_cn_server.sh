#!/usr/bin/env bash
# 在仅国内镜像可达的服务器上：用 pip 装 uv（不经 astral.sh），再 uv sync。
# 用法：bash scripts/bootstrap_cn_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] uv 未找到，使用 pip 从镜像安装: $INDEX"
  if command -v python3 >/dev/null 2>&1; then
    python3 -m pip install -U uv -i "$INDEX"
  elif [[ -x "${HOME}/miniconda3/bin/pip" ]]; then
    "${HOME}/miniconda3/bin/pip" install -U uv -i "$INDEX"
    export PATH="${HOME}/miniconda3/bin:${PATH}"
  else
    echo "[bootstrap] 请安装 Python3 pip 或 Miniconda 后重试。" >&2
    exit 1
  fi
fi

export UV_DEFAULT_INDEX="$INDEX"

echo "[bootstrap] uv sync（train-cu + dev）…"
uv sync --extra dev --extra train-cu

echo "[bootstrap] 完成。可运行: uv run python scripts/run_dltr.py --help"
