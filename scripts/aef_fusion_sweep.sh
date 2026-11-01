#!/usr/bin/env bash
# Run the three AEF fusion variants (none / real / shuffled) once the GPU is idle.
#
# Diagnostic only: tile-level AP, one seed, ~2k steps. It waits for the card to go
# below WAIT_MB so it can be queued behind the formal T9/T10 lanes without
# touching them.
#
#   bash scripts/aef_fusion_sweep.sh --detached      # queue and return
#   GPU_WAIT=0 bash scripts/aef_fusion_sweep.sh      # start now
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STEPS="${AEF_STEPS:-2000}"
TRAIN_IMAGES="${AEF_TRAIN_IMAGES:-40}"
VAL_IMAGES="${AEF_VAL_IMAGES:-8}"
FEATURES="${AEF_FEATURES:-/home/yzhong/rs-eval/aef-features}"
OUT="${AEF_OUT:-/home/yzhong/rs-eval/aef-fusion}"
GPU_WAIT="${GPU_WAIT:-1}"
WAIT_MB="${WAIT_MB:-1500}"
PY="${PY:-workers/lim/.venv/bin/python}"
LOGDIR="${LOGDIR:-/home/yzhong/rs-eval/logs-isaid}"

if [ "${1:-}" = "--detached" ]; then
    mkdir -p "$LOGDIR"
    log="$LOGDIR/aef-fusion-$(date +%Y%m%d-%H%M%S).log"
    setsid nohup env GPU_WAIT="$GPU_WAIT" AEF_STEPS="$STEPS" AEF_TRAIN_IMAGES="$TRAIN_IMAGES" \
        AEF_VAL_IMAGES="$VAL_IMAGES" AEF_FEATURES="$FEATURES" AEF_OUT="$OUT" \
        bash "$0" > "$log" 2>&1 < /dev/null &
    echo "$(date -Is) detached; log $log"
    exit 0
fi

if [ "$GPU_WAIT" = "1" ]; then
    while :; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ -z "$used" ] || [ "$used" -lt "$WAIT_MB" ]; then
            break
        fi
        echo "$(date -Is) GPU busy (${used} MiB); waiting"
        sleep 300
    done
fi

mkdir -p "$OUT"
for variant in none real shuffled; do
    echo "$(date -Is) START $variant"
    PYTHONPATH=src "$PY" scripts/aef_fusion_probe.py \
        --variant "$variant" --steps "$STEPS" \
        --train-images "$TRAIN_IMAGES" --val-images "$VAL_IMAGES" \
        --features "$FEATURES" --out "$OUT" || echo "$(date -Is) FAIL $variant"
done
echo "$(date -Is) === AEF fusion sweep done ==="
