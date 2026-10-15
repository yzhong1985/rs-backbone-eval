#!/usr/bin/env bash
# Re-run baseline cells under the current protocol (see docs/reviews/历史运行审计.md).
#
# Historical runs predate the value-changing fixes or carry no recorded code
# version, so they cannot be reused as comparisons. This re-runs the chosen
# cells with new run ids ("-v3") and never overwrites the originals. It waits
# for the GPU, runs sequentially, and exits non-zero if any run failed.
#
# Usage: scripts/rerun_baselines.sh T1|T1A|T5
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
# AnySat's adapter path sits right at the 24 GB limit; expandable segments avoids
# fragmentation OOMs without changing the effective batch (fairness).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASK="$(printf '%s' "${1:-T1}" | tr '[:lower:]' '[:upper:]')"
LOGDIR="${RS_EVAL_RERUN_LOGDIR:-/tmp/opencode}"
if ! mkdir -p "$LOGDIR"; then
    echo "cannot create log directory: $LOGDIR" >&2
    exit 1
fi
LOG="$LOGDIR/rerun-${TASK}-baselines.out"
exec > "$LOG" 2>&1

echo "$(date -Is) === rerun $TASK baselines (logs in $LOGDIR) ==="
while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$used" ]; then echo "$(date -Is) no GPU visible"; exit 1; fi
    if [ "$used" -le 2000 ]; then break; fi
    echo "$(date -Is) GPU busy (${used} MiB); waiting"
    sleep 300
done
echo "$(date -Is) GPU free; starting"

failed=0
succeeded=0

run() {  # model protocol seed
    local model=$1 proto=$2 seed=$3
    local rid="${model}-${TASK}-${proto}-${seed}-${RS_EVAL_RERUN_SUFFIX:-v3}"
    local rlog="$LOGDIR/${rid}.log"
    echo "$(date -Is) START $rid"
    if uv run rs-eval dispatch --action train --task "$TASK" --model "$model" \
        --protocol "$proto" --seed "$seed" --run-id "$rid" > "$rlog" 2>&1; then
        succeeded=$((succeeded + 1))
        echo "$(date -Is) OK $rid $(grep -oE 'water_iou=[0-9.]+|mean_iou=[0-9.]+' "$rlog" | tail -1)"
    else
        failed=$((failed + 1))
        echo "$(date -Is) FAIL $rid rc=$? (see $rlog)"
    fi
}

run_anysat() {  # protocol seed [micro_batch] [grad_accum]
    # AnySat's T1 adapter path OOMs at the official batch of 2; a micro-batch of
    # 1 with two accumulation steps keeps the *effective* batch (and fairness)
    # unchanged. Callers can override for cells where the full batch fits.
    export ANYSAT_WORKER_BATCH_SIZE="${3:-1}"
    export ANYSAT_WORKER_GRAD_ACCUM="${4:-2}"
    run anysat_base "$1" "$2"
    unset ANYSAT_WORKER_BATCH_SIZE ANYSAT_WORKER_GRAD_ACCUM
}

run_explicit() {  # run_id action protocol seed
    # Same dispatch/log bookkeeping as run() but with a fully explicit run id,
    # for reference cells whose historical ids carry lr variants or actions.
    local rid=$1 action=$2 proto=$3 seed=$4
    local rlog="$LOGDIR/${rid}.log"
    echo "$(date -Is) START $rid"
    if uv run rs-eval dispatch --action "$action" --task "$TASK" --model prithvi_eo_v2_300_tl \
        --protocol "$proto" --seed "$seed" --run-id "$rid" > "$rlog" 2>&1; then
        succeeded=$((succeeded + 1))
        echo "$(date -Is) OK $rid $(grep -oE 'water_iou=[0-9.]+|mean_iou=[0-9.]+' "$rlog" | tail -1)"
    else
        failed=$((failed + 1))
        echo "$(date -Is) FAIL $rid rc=$? (see $rlog)"
    fi
}

case "$TASK" in
    T1)
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl frozen_probe "$seed"; done
        for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        for seed in 17 42 73; do run_anysat frozen_adapter "$seed"; done
        ;;
    T1A)
        TASK=T1   # only the AnySat cell of the T1 frozen baselines
        for seed in 17 42 73; do run_anysat frozen_adapter "$seed"; done
        ;;
    T5)
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl frozen_probe "$seed"; done
        for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        for seed in 17 42 73; do run_anysat frozen_probe "$seed"; done
        ;;
    O)
        # OlmoEarth T1 official seeds: the worker only accepts frozen_adapter
        # (the six-band constant-fill adapter, report identity = protocol_id).
        TASK=T1  # queue letter stays O (log name); dispatch target is T1
        for seed in 17 42 73; do run olmoearth_v1_2_tiny frozen_adapter "$seed"; done
        ;;
    T5A)
        # AnySat T5 frozen cell only. The batch-2 memory probe OOMed (23.0GB in
        # use, 476MB free) while 1x2 accumulation peaks ~20.4GB, so the cell
        # keeps the accumulation setting; cost is ~17 min/epoch (125 official
        # epochs). Set a fresh suffix if -v3 ids are taken (aborted probe seed
        # holds one).
        TASK=T5  # queue letter stays T5A (log name); dispatch target is T5
        for seed in 17 42 73; do run_anysat frozen_probe "$seed"; done
        ;;
    B0)
        # Prithvi T1 reference reproduction: the paper protocol sweep (official
        # lr plus the 1e-4 / 2e-4 variants at three seeds) and the repo-config
        # run plus its probe. The historical 11 records carry no code version
        # and predate the restore/selection fix, so they are re-runs with
        # explicit ids and honors RS_EVAL_RERUN_SUFFIX.
        TASK=T1  # queue letter stays B0 (log name); dispatch target is T1
        SUFFIX="${RS_EVAL_RERUN_SUFFIX:-v3}"
        for seed in 17 42 73; do
            unset PRITHVI_WORKER_LR  # official lr from the worker default
            run_explicit \
                "prithvi_eo_v2_300_tl-T1-prithvi_paper_protocol-train-${seed}-${SUFFIX}" \
                train prithvi_paper_protocol "$seed"
            unset PRITHVI_WORKER_LR
            PRITHVI_WORKER_LR=1e-4 run_explicit \
                "prithvi_eo_v2_300_tl-T1-prithvi_paper_protocol-train-${seed}-lr1e4-${SUFFIX}" \
                train prithvi_paper_protocol "$seed"
            PRITHVI_WORKER_LR=2e-4 run_explicit \
                "prithvi_eo_v2_300_tl-T1-prithvi_paper_protocol-train-${seed}-lr2e4-${SUFFIX}" \
                train prithvi_paper_protocol "$seed"
            unset PRITHVI_WORKER_LR
        done
        run_explicit "prithvi_eo_v2_300_tl-T1-prithvi_repo_config-train-42-${SUFFIX}" \
            train prithvi_repo_config 42
        run_explicit "prithvi_eo_v2_300_tl-T1-prithvi_repo_config-probe-42-${SUFFIX}" \
            probe prithvi_repo_config 42
        ;;
    B1)
        # T1 full-finetune parity cells (audit: highest-priority stale B1 cells).
        # AnySat keeps micro-batch accumulation so the effective batch and the
        # 19.3GB peak stay compatible with the frozen-track conventions.
        TASK=T1  # queue letter stays B1 (log name); dispatch target is T1
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl full_finetune "$seed"; done
        for seed in 17 42 73; do run clay_v1_5 full_finetune "$seed"; done
        for seed in 17 42 73; do run_anysat full_finetune "$seed"; done
        ;;
    *)
        echo "unknown task: $TASK"; exit 2 ;;
esac

echo "$(date -Is) === $TASK done: $succeeded ok, $failed failed ==="
if [ "$failed" -gt 0 ]; then
    echo "$(date -Is) FAILURES: $failed run(s) failed" >&2
    exit 1
fi
exit 0
