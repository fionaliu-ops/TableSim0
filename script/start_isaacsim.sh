#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ISAACSIM_ROOT="${TABLESIM_ISAACSIM_ROOT:-/home/ubuntu/isaacsim}"
TARGET_CONTAINER="${TABLESIM_ISAACSIM_CONTAINER:-tablesim-isaac-sim}"
TMP_SCRIPT="/tmp/tablesim_smoke_isaacsim.py"

if [ -x "${ISAACSIM_ROOT}/python.sh" ]; then
  "${ISAACSIM_ROOT}/python.sh" "${ROOT_DIR}/script/smoke_isaacsim.py"
  exit 0
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${TARGET_CONTAINER}"; then
  echo "Isaac Sim container not running: ${TARGET_CONTAINER}" >&2
  echo "Also did not find a usable Isaac Sim install at ${ISAACSIM_ROOT}" >&2
  exit 1
fi

docker cp "${ROOT_DIR}/script/smoke_isaacsim.py" "${TARGET_CONTAINER}:${TMP_SCRIPT}"
docker exec "${TARGET_CONTAINER}" bash -lc '
  if command -v python3 >/dev/null 2>&1; then
    python3 /tmp/tablesim_smoke_isaacsim.py
  elif command -v python >/dev/null 2>&1; then
    python /tmp/tablesim_smoke_isaacsim.py
  elif [ -x /isaac-sim/python.sh ]; then
    /isaac-sim/python.sh /tmp/tablesim_smoke_isaacsim.py
  elif [ -x /root/.local/share/ov/pkg/isaac-sim/python.sh ]; then
    /root/.local/share/ov/pkg/isaac-sim/python.sh /tmp/tablesim_smoke_isaacsim.py
  else
    echo "No usable Python entrypoint found in Isaac Sim container" >&2
    exit 2
  fi
'
