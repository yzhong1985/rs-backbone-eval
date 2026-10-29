#!/usr/bin/env bash
# Two-point throughput measurement for a training worker.
#
# Why two runs: the worker reports `train_tiles_per_second` over the whole
# training loop, so a short run is dominated by cuDNN warmup and allocation
# overhead. Both runs below start cold, so the same warmup is present in each
# and the *difference* isolates the steady-state cost:
#
#   steady_seconds_per_step = (t_short_steps+steady - t_short) / steady_steps
#
# with t = tiles / tiles_per_second from the run's own tags. Nothing in the
# worker changes, so the measurement cannot drift from the training code.
#
# It waits for the GPU to be idle first: a measurement taken next to a formal
# run measures contention, not the worker (2026-10-29: 0.71 tile/s beside
# Prithvi instead of the isolated number).
#
# Usage:
#   bash scripts/measure_throughput.sh --task T10 --model lim_mit_huge \
#       --protocol rotated_frozen --prefix LIM_WORKER --micro 1 --accum 2 \
#       --manifest /geodata/rs-eval/data/manifests/T10/holdout_1024_s824_seed17.jsonl \
#       --suffix thr1
set -euo pipefail

TASK="" MODEL="" PROTOCOL="" PREFIX="" MICRO="1" ACCUM="1" MANIFEST=""
SUFFIX="thr" WARMUP=100 STEADY=200 GPU_LIMIT=2000 RUNS_ROOT="${RS_EVAL_RUNS_ROOT:-/geodata/rs-eval/runs}"
LOGDIR="${RS_EVAL_MEASURE_LOGDIR:-/home/yzhong/rs-eval/logs-isaid}"

while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --protocol) PROTOCOL=$2; shift 2 ;;
        --prefix) PREFIX=$2; shift 2 ;;
        --micro) MICRO=$2; shift 2 ;;
        --accum) ACCUM=$2; shift 2 ;;
        --manifest) MANIFEST=$2; shift 2 ;;
        --suffix) SUFFIX=$2; shift 2 ;;
        --warmup) WARMUP=$2; shift 2 ;;
        --steady) STEADY=$2; shift 2 ;;
        --runs-root) RUNS_ROOT=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for required in TASK MODEL PROTOCOL PREFIX; do
    if [ -z "${!required}" ]; then echo "missing --${required,,}" >&2; exit 2; fi
done

if [ -n "$MANIFEST" ]; then export RS_EVAL_MANIFEST="$MANIFEST"; fi
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/throughput-${MODEL}-${TASK}-${SUFFIX}.txt"

echo "$(date -Is) waiting for <= ${GPU_LIMIT} MiB used on the GPU"
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$used" ]; then echo "$(date -Is) no GPU visible" >&2; exit 1; fi
    if [ "$used" -le "$GPU_LIMIT" ]; then break; fi
    echo "$(date -Is) GPU busy (${used} MiB); waiting"
    sleep 300
done
echo "$(date -Is) GPU free; measuring"

measure() {  # batches -> prints "tiles tiles_per_second"
    local batches=$1
    local rid="${MODEL}-${TASK}-${PROTOCOL}-${SUFFIX}-b${batches}"
    local log="$LOGDIR/${rid}.log"
    echo "$(date -Is) START $rid (${batches} micro batches)" >&2
    env "${PREFIX}_EPOCHS=1" "${PREFIX}_LIMIT_BATCHES=${batches}" \
        "${PREFIX}_EVAL_IMAGES=1" "${PREFIX}_NUM_WORKERS=4" \
        uv run rs-eval dispatch --action train --task "$TASK" --model "$MODEL" \
        --protocol "$PROTOCOL" --seed 17 --run-id "$rid" >"$log" 2>&1
    python3 - "$RUNS_ROOT" "$rid" <<'PY'
import json, sys
from pathlib import Path
runs_root, rid = sys.argv[1], sys.argv[2]
record = json.loads((Path(runs_root) / rid / "run.json").read_text())
tags = record["tags"]
tiles = int(tags["train_tiles"])
rate = float(tags["train_tiles_per_second"])
peak = record["training"].get("peak_gpu_memory_mb") or 0.0
print(f"{tiles} {rate} {peak:.0f}")
PY
}

read -r tiles_short rate_short peak_short < <(measure "$((WARMUP * ACCUM))")
read -r tiles_long rate_long peak_long < <(measure "$(((WARMUP + STEADY) * ACCUM))")

python3 - "$SUMMARY" "$TASK" "$MODEL" "$PROTOCOL" "$MICRO" "$ACCUM" \
    "$tiles_short" "$rate_short" "$tiles_long" "$rate_long" "$peak_long" <<'PY'
import sys
from pathlib import Path

(out, task, model, protocol, micro, accum,
 tiles_short, rate_short, tiles_long, rate_long, peak) = sys.argv[1:]
tiles_short, rate_short, tiles_long, rate_long = map(float, (tiles_short, rate_short, tiles_long, rate_long))
steps_short = tiles_short / float(accum)
steps_steady = (tiles_long - tiles_short) / float(accum)
seconds_short = tiles_short / rate_short
seconds_long = tiles_long / rate_long
steady_per_step = (seconds_long - seconds_short) / steps_steady
steady_tiles = float(accum) / steady_per_step
conservative = tiles_long / seconds_long

lines = [
    f"task={task} model={model} protocol={protocol} micro={micro} accum={accum}",
    f"cold:          {tiles_short:.0f} tiles, {steps_short:.0f} steps, {seconds_short:.1f} s, {rate_short:.2f} tile/s",
    f"warm+steady:   {tiles_long:.0f} tiles, {tiles_long / float(accum):.0f} steps, {seconds_long:.1f} s, {rate_long:.2f} tile/s",
    f"steady state:  {steady_per_step:.3f} s per optimizer step, {steady_tiles:.2f} tile/s",
    f"whole-run rate (conservative, includes warmup): {conservative:.2f} tile/s",
    f"peak_gpu_memory_mb={peak}",
    f"projected wall time per 1,000 optimizer steps: {steady_per_step * 1000 / 3600:.2f} h",
    f"projected wall time per 10,000 optimizer steps: {steady_per_step * 10000 / 3600:.2f} h",
    f"projected wall time per 25,000 optimizer steps: {steady_per_step * 25000 / 3600:.2f} h",
]
text = "\n".join(lines)
Path(out).write_text(text + "\n")
print(text)
PY

echo "$(date -Is) wrote $SUMMARY"
