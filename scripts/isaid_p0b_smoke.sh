#!/usr/bin/env bash
# P0-b throughput smoke for the iSAID / LIM branch (design 2026-10-29 §5.3).
#
# Measures the two budgets that matter before scheduling real runs:
#   * B2U (upernet_frozen) vs B1U (upernet_finetune), at 512 and 1024 px;
#   * the whole-image sliding-window evaluation cost.
#
# This is *not a result*: every run is one short epoch, carries [SMOKE] in its
# summary, and must never be used as a headline number.
#
# Usage:
#   bash scripts/isaid_p0b_smoke.sh            # waits for the GPU, then runs
#   SKIP_WAIT=1 bash scripts/isaid_p0b_smoke.sh
#
# Environment: RS_EVAL_MANIFEST is set explicitly per geometry so the run does
# not depend on which manifest happens to be newest.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BATCHES=${BATCHES:-20}
EPOCHS=${EPOCHS:-1}
EVAL_IMAGES=${EVAL_IMAGES:-3}
NORM_SAMPLES=${NORM_SAMPLES:-8}
NUM_WORKERS=${NUM_WORKERS:-2}
SUFFIX=${SUFFIX:-smoke}
WAIT_SECONDS=${WAIT_SECONDS:-10800}

log_root=${LOG_ROOT:-/tmp/opencode}
mkdir -p "$log_root"
log="$log_root/isaid-p0b-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$log") 2>&1
echo "log: $log"

data_root=$(uv run python -c 'from rs_backbone_eval.paths import RuntimePaths; print(RuntimePaths.from_environment().data_root)')
manifest_dir="$data_root/manifests/T9"
echo "data root: $data_root"

if [ "${SKIP_WAIT:-0}" != "1" ]; then
  echo "waiting for the GPU to be free (up to ${WAIT_SECONDS}s)..."
  waited=0
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do
    if [ "$waited" -ge "$WAIT_SECONDS" ]; then
      echo "GPU still busy after ${WAIT_SECONDS}s; aborting" >&2
      exit 1
    fi
    sleep 60
    waited=$((waited + 60))
  done
  echo "GPU free after ${waited}s"
fi

# build_geometry <crop> <stride> -> prints the pinned holdout manifest path
build_geometry() {
  local crop=$1 stride=$2
  local name="holdout_${crop}_s${stride}_seed17.jsonl"
  RS_EVAL_ISAID_CROP="$crop" RS_EVAL_ISAID_STRIDE="$stride" \
    uv run rs-eval build-manifest --task T9 >/dev/null
  local base
  base=$(ls -t "$manifest_dir"/*.jsonl | grep -v '/holdout_' | head -1)
  uv run rs-eval holdout-manifest --manifest "$base" --out "$manifest_dir/$name" \
    --fraction 0.2 --seed 17 --split validation >/dev/null
  echo "$manifest_dir/$name"
}

smoke() {
  local protocol=$1 manifest=$2 tag=$3
  local run_id="lim_mit_huge-T9-$protocol-train-17-$SUFFIX-$tag"
  echo "=== $tag ($protocol, manifest $(basename "$manifest"))"
  RS_EVAL_MANIFEST="$manifest" \
    LIM_WORKER_EPOCHS="$EPOCHS" \
    LIM_WORKER_LIMIT_BATCHES="$BATCHES" \
    LIM_WORKER_EVAL_IMAGES="$EVAL_IMAGES" \
    LIM_WORKER_NORM_SAMPLES="$NORM_SAMPLES" \
    LIM_WORKER_NUM_WORKERS="$NUM_WORKERS" \
    uv run rs-eval dispatch --action train --task T9 --model lim_mit_huge \
      --protocol "$protocol" --seed 17 --run-id "$run_id" || echo "!! $tag failed"
}

manifest_1024=$(build_geometry 1024 512)
manifest_512=$(build_geometry 512 256)

echo "=== probe (1024)"
RS_EVAL_MANIFEST="$manifest_1024" uv run rs-eval dispatch --action probe --task T9 \
  --model lim_mit_huge --protocol upernet_frozen --seed 17 \
  --run-id "lim_mit_huge-T9-upernet_frozen-probe-17-$SUFFIX" || echo "!! probe failed"

smoke upernet_frozen "$manifest_1024" 1024-frozen
smoke upernet_finetune "$manifest_1024" 1024-finetune
smoke upernet_frozen "$manifest_512" 512-frozen
smoke upernet_finetune "$manifest_512" 512-finetune

echo
echo "=== summary (from run.json; SMOKE runs are not results)"
uv run python - "$SUFFIX" <<'PY'
import json
import sys
from pathlib import Path

from rs_backbone_eval.paths import RuntimePaths

suffix = sys.argv[1]
runs = RuntimePaths.from_environment().runs_root
rows = []
for path in sorted(runs.glob("lim_mit_huge-T9-*")):
    record = path / "run.json"
    if not record.is_file() or suffix not in path.name:
        continue
    data = json.loads(record.read_text(encoding="utf-8"))
    tags = data.get("tags", {})
    rows.append(
        (
            path.name,
            data.get("status"),
            tags.get("train_crop_px", "-"),
            tags.get("backbone_frozen", "-"),
            tags.get("train_tiles_per_second", "-"),
            tags.get("eval_seconds_per_image", "-"),
            tags.get("peak_gpu_mb", "-"),
            (data.get("metrics") or [{}])[0].get("value") if data.get("metrics") else "-",
        )
    )
head = ("run", "status", "crop", "frozen", "tiles/s", "eval s/img", "peak MB", "metric")
print(" | ".join(head))
for row in rows:
    print(" | ".join(str(value) for value in row))
PY
echo "log: $log"
