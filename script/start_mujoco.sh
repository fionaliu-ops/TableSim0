#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm \
  -e MUJOCO_GL=egl \
  -v "${ROOT_DIR}:/workspace/TableSim" \
  -w /workspace/TableSim \
  duo-teleop-eval:latest \
  python3 script/smoke_mujoco.py
