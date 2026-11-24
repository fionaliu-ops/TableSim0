#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="${1:?Usage: bash collect_data.sh TASK_NAME [TASK_CONFIG] [GPU_ID]}"
TASK_CONFIG="${2:-${TASK_NAME}}"
GPU_ID="${3:-0}"

cd "${ROOT_DIR}"

if [ -f /.dockerenv ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  exec python script/collect_data.py "${TASK_NAME}" "${TASK_CONFIG}"
fi

exec docker compose run --rm \
  -e CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  tablesim \
  python script/collect_data.py "${TASK_NAME}" "${TASK_CONFIG}"
