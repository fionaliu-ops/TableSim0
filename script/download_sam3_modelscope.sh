#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_REVISION="f8ee5d16d5c30d0e87b1d22cbd358ce97761df56"
MODEL_URL="https://www.modelscope.cn/models/facebook/sam3/resolve/${MODEL_REVISION}/sam3.pt"
EXPECTED_SHA256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
EXPECTED_SIZE="3450062241"
CHECKPOINT_PATH="/home/tablesim/.cache/huggingface/modelscope/facebook/sam3/sam3.pt"

cd "${ROOT_DIR}"
docker compose --profile sam3 up -d --no-deps sam3

docker compose exec -T sam3 python -u - \
  "${MODEL_URL}" \
  "${EXPECTED_SHA256}" \
  "${EXPECTED_SIZE}" \
  "${CHECKPOINT_PATH}" <<'PY'
import hashlib
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen


url, expected_sha256, expected_size_text, checkpoint_text = sys.argv[1:]
expected_size = int(expected_size_text)
checkpoint = Path(checkpoint_text)
partial = checkpoint.with_suffix(checkpoint.suffix + ".part")
checkpoint.parent.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if checkpoint.is_file() and checkpoint.stat().st_size == expected_size:
    print("Existing checkpoint found; verifying SHA-256.", flush=True)
    if sha256(checkpoint) == expected_sha256:
        print(f"SAM3 checkpoint is ready: {checkpoint}", flush=True)
        raise SystemExit(0)
    checkpoint.unlink()

start = partial.stat().st_size if partial.exists() else 0
headers = {"User-Agent": "TableSim-SAM3-ModelScope/1.0"}
if start:
    headers["Range"] = f"bytes={start}-"

request = Request(url, headers=headers)
with urlopen(request, timeout=120) as response:
    status = getattr(response, "status", response.getcode())
    resumed = bool(start and status == 206)
    if not resumed:
        start = 0
    mode = "ab" if resumed else "wb"
    downloaded = start
    report_step = 256 * 1024 * 1024
    next_report = ((downloaded // report_step) + 1) * report_step
    print(f"Downloading from byte {downloaded}.", flush=True)
    with partial.open(mode) as output:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                print(f"Downloaded {downloaded}/{expected_size} bytes.", flush=True)
                next_report += report_step

if partial.stat().st_size != expected_size:
    raise SystemExit(
        f"Checkpoint size mismatch: expected {expected_size}, "
        f"got {partial.stat().st_size}. Re-run this script to resume."
    )

print("Download complete; verifying SHA-256.", flush=True)
actual_sha256 = sha256(partial)
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, "
        f"got {actual_sha256}."
    )

os.replace(partial, checkpoint)
print(f"SAM3 checkpoint is ready: {checkpoint}", flush=True)
PY
