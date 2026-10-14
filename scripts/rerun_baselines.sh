#!/usr/bin/env bash
# Re-run baseline cells under the current protocol (see docs/历史运行审计.md).
#
# Historical runs predate the P0 seed/adapter fix, the T5/R6 full-label grid, or
# carry no recorded code version, so they cannot be reused as comparisons. This
# re-runs the chosen cells with new run ids ("-v3") and never overwrites the
# originals. It waits for the GPU to be free, then runs sequentially.
#
# Usage: scripts/rerun_baselines.sh t1     # T1 B2 frozen, three models x 3 seeds
set -uo pipefail
cd "$(dirname "$0")/.."

# The project's runtime paths and checkpoints live in .env; uv does not load it
# automatically, so source it explicitly.
set -a
# shellcheck disable=SC1091
. ./.env
set +a
export PATH="$HOME/.local/bin:$PATH"
export http_proxy="${RS_EVAL_PROXY:-}"
export https_proxy="${RS_EVAL_PROXY:-}"

TASK="$(printf '%s' "${1:-T1}" | tr '[:lower:]' '[:upper:]')"
LOG="/tmp/opencode/rerun-${TASK}-baselines.out"
exec > "$LOG" 2>&1

echo "$(date -Is) === rerun $TASK baselines ==="
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$used" ]; then echo "$(date -Is) no GPU visible"; exit 1; fi
    if [ "$used" -le 2000 ]; then break; fi
    echo "$(date -Is) GPU busy (${used} MiB); waiting"
    sleep 300
done
echo "$(date -Is) GPU free; starting"

run() {  # model protocol seed
    local model=$1 proto=$2 seed=$3
    local rid="${model}-${TASK}-${proto}-${seed}-v3"
    echo "$(date -Is) START $rid"
    uv run rs-eval dispatch --action train --task "$TASK" --model "$model" \
        --protocol "$proto" --seed "$seed" --run-id "$rid" > "/tmp/opencode/${rid}.log" 2>&1
    local rc=$?
    echo "$(date -Is) END $rid rc=$rc $(grep -oE 'water_iou=[0-9.]+|mean_iou=[0-9.]+' "/tmp/opencode/${rid}.log" | tail -1)"
}

case "$TASK" in
    T1)
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl frozen_probe "$seed"; done
        for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        for seed in 17 42 73; do run anysat_base frozen_adapter "$seed"; done
        ;;
    T5)
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl frozen_probe "$seed"; done
        for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        for seed in 17 42 73; do run anysat_base frozen_probe "$seed"; done
        ;;
    *)
        echo "unknown task: $TASK"; exit 2 ;;
esac
echo "$(date -Is) === $TASK BASELINES DONE ==="
