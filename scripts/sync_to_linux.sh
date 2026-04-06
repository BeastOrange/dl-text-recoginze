#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f "PLAN.md" ]]; then
  echo "ERROR: Must run inside project repository containing PLAN.md"
  exit 1
fi

if [[ -z "${REMOTE:-}" ]]; then
  echo "ERROR: REMOTE is required. Example: user@server"
  exit 1
fi

if [[ -z "${TARGET_DIR:-}" ]]; then
  echo "ERROR: TARGET_DIR is required. Example: /home/user/dl-text-recoginze"
  exit 1
fi

RSYNC_ARGS=(
  -az
  --delete
  --exclude=.git/
  --exclude=.venv/
  --exclude=.pytest_cache/
  --exclude=.ruff_cache/
  --exclude=__pycache__/
  --exclude=data/raw/
  --exclude=data/interim/
  --exclude=data/processed/
  --exclude=artifacts/cache/
  --exclude=artifacts/checkpoints/
  --exclude=artifacts/tmp/
  --exclude=*.pt
  --exclude=*.pth
  --exclude=*.onnx
  --exclude=*.ckpt
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run -v)
fi

echo "Syncing project to ${REMOTE}:${TARGET_DIR}"
rsync "${RSYNC_ARGS[@]}" "${ROOT_DIR}/" "${REMOTE}:${TARGET_DIR}/"
echo "Sync complete."
