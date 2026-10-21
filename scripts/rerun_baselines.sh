#!/usr/bin/env bash
# Re-run baseline cells under the current protocol (see docs/reviews/历史运行审计.md).
#
# Historical runs predate the value-changing fixes or carry no recorded code
# version, so they cannot be reused as comparisons. This re-runs the chosen
# cells with new run ids ("-v3") and never overwrites the originals. It waits
# for the GPU, runs sequentially, and exits non-zero if any run failed.
#
# Usage: scripts/rerun_baselines.sh T1|T1A|T5
#        scripts/rerun_baselines.sh --detached T2A   # survives session teardown
#
# Long chains MUST run with --detached: the script re-execs under `setsid`
# (its own session, no controlling tty) so a SIGHUP from the launching shell
# being reaped never reaches the worker mid-epoch. T2A -v5/-v6 died exactly
# this way — the worker reached epoch 40 then vanished with no traceback, no
# OOM, no GPU Xid; the run.json stayed "running" because writer.complete()
# never ran. The detached PID lands in $LOGDIR/rerun-<TASK>-baselines.pid;
# stop the chain with `kill -- -$(cat $PIDFILE)` (process-group signal).
set -uo pipefail
cd "$(dirname "$0")/.."

# The project's runtime paths and checkpoints live in .env; uv does not load it
# automatically, so source it explicitly. An explicitly exported
# RS_EVAL_RUNS_ROOT wins over .env so the completed-run guard below can be
# pointed at an isolated directory (tests), while production keeps using .env.
_runs_root_override="${RS_EVAL_RUNS_ROOT:-}"
set -a
# shellcheck disable=SC1091
. ./.env
set +a
if [ -n "$_runs_root_override" ]; then
    export RS_EVAL_RUNS_ROOT="$_runs_root_override"
fi
export PATH="$HOME/.local/bin:$PATH"
export http_proxy="${RS_EVAL_PROXY:-}"
export https_proxy="${RS_EVAL_PROXY:-}"
# AnySat's adapter path sits right at the 24 GB limit; expandable segments avoids
# fragmentation OOMs without changing the effective batch (fairness).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASK=""
DETACH=0
case "${1:-}" in
    --detached)
        DETACH=1
        shift
        TASK="$(printf '%s' "${1:-T1}" | tr '[:lower:]' '[:upper:]')"
        ;;
    "") ;;
    *) TASK="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')" ;;
esac
TASK="${TASK:-T1}"
LOGDIR="${RS_EVAL_RERUN_LOGDIR:-/tmp/opencode}"
if ! mkdir -p "$LOGDIR"; then
    echo "cannot create log directory: $LOGDIR" >&2
    exit 1
fi
LOG="$LOGDIR/rerun-${TASK}-baselines.out"
PIDFILE="$LOGDIR/rerun-${TASK}-baselines.pid"

# --detached: fork into a new session exactly once, then run inline. The env
# guard breaks the recursion so the re-exec'd child skips this branch.
if [ "$DETACH" = 1 ] && [ -z "${RS_EVAL_RERUN_DETACHED:-}" ]; then
    export RS_EVAL_RERUN_DETACHED=1
    RS_EVAL_RERUN_READY_FILE=$(mktemp "$LOGDIR/.rerun-${TASK}-ready.XXXXXX") || exit 1
    export RS_EVAL_RERUN_READY_FILE
    # setsid --fork always forks a new session leader; stdin from /dev/null so a
    # tty hangup can never reach the chain. The launcher only waits for the
    # child to publish its own PID (setsid's $! is the wrapper).
    setsid --fork bash "$0" --detached "$TASK" </dev/null >> "$LOG" 2>&1 &
    # $! is the short-lived setsid parent, not the detached session leader.
    # Let the child publish its own PID, then acknowledge that exact process.
    for ((attempt = 0; attempt < 100; attempt++)); do
        [ -s "$RS_EVAL_RERUN_READY_FILE" ] && break
        sleep 0.05
    done
    if [ ! -s "$RS_EVAL_RERUN_READY_FILE" ]; then
        rm -f "$RS_EVAL_RERUN_READY_FILE"
        echo "detached child did not become ready; see $LOG" >&2
        exit 1
    fi
    PID=$(cat "$RS_EVAL_RERUN_READY_FILE")
    rm -f "$RS_EVAL_RERUN_READY_FILE"
    echo "$(date -Is) detached; PID $PID (pidfile $PIDFILE); log $LOG"
    exit 0
fi

exec > "$LOG" 2>&1
# Clean the PID file on exit (detached or not); leave the log for review.
trap 'rm -f "$PIDFILE"' EXIT
# In detached mode this is the session leader, not the setsid wrapper.
echo "$$" > "$PIDFILE" || exit 1
if [ "$DETACH" = 1 ]; then
    if [ -n "${RS_EVAL_RERUN_READY_FILE:-}" ]; then
        echo "$$" > "$RS_EVAL_RERUN_READY_FILE" || exit 1
        unset RS_EVAL_RERUN_READY_FILE
    fi
fi

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
    # Never clobber a completed run: a shared suffix (the default v3) re-run
    # would silently overwrite a finished run.json. Bump RS_EVAL_RERUN_SUFFIX.
    local runs_root="${RS_EVAL_RUNS_ROOT:-/geodata/rs-eval/runs}"
    if [ -f "$runs_root/$rid/run.json" ] && \
        grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$runs_root/$rid/run.json" 2>/dev/null; then
        echo "$(date -Is) SKIP $rid: a completed run exists (bump RS_EVAL_RERUN_SUFFIX)"
        failed=$((failed + 1))
        return
    fi
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
    T5B)
        # Same cell as T5A but a single seed: the 4090 skips the seed handed to
        # the Colab A100 queue, so its own list shrinks to the remaining seeds.
        TASK=T5  # queue letter stays T5B (log name); dispatch target is T5
        for seed in 73; do run_anysat frozen_probe "$seed"; done
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
    T2)
        # T2 frozen baselines (fire scars): three backbones x 3 seeds.
        # AnySat keeps its historical frozen_adapter protocol; batch 1 x 2 fits
        # the ~23.4GB peak on a 24GB card.
        TASK=T2  # queue letter equals the dispatch task here
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl frozen_probe "$seed"; done
        for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        for seed in 17 42 73; do run_anysat frozen_adapter "$seed"; done
        ;;
    T2A)
        # Only the AnySat T2 frozen cell (restart after an operator cleanup).
        TASK=T2  # queue letter stays T2A (log name); dispatch target is T2
        for seed in 17 42 73; do run_anysat frozen_adapter "$seed"; done
        ;;
    B1)
        # T1 full-finetune parity cells on the cheap GPUs (prithvi/clay).
        # AnySat needs ~23.4GB per micro-batch: it runs under case B1A on a
        # big-memory GPU (Colab A100 / another queue letter) instead.
        TASK=T1  # queue letter stays B1 (log name); dispatch target is T1
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl full_finetune "$seed"; done
        for seed in 17 42 73; do run clay_v1_5 full_finetune "$seed"; done
        ;;
    B1A)
        # AnySat T1 full-finetune cell alone: needs >24GB for the native batch 2,
        # so queue it on an A100/H100-class GPU (Colab) or after T5A on the 4090.
        TASK=T1  # queue letter stays B1A (log name); dispatch target is T1
        # On 24GB cards the native batch 2 OOMs: set B1A_MICRO=1 B1A_ACC=2 there.
        for seed in 17 42 73; do run_anysat full_finetune "$seed" "${B1A_MICRO:-2}" "${B1A_ACC:-1}"; done
        ;;
    T3C)
        # AnySat T3 full-finetune, single seed (the cell's protocol budget).
        TASK=T3  # queue letter stays T3C (log name); dispatch target is T3
        run_anysat full_finetune 42 "${T3C_MICRO:-1}" "${T3C_ACC:-2}"
        ;;
    T4C)
        # T4 full-finetune cells (landslide4sense): prithvi + anysat, one seed.
        # The anyst T4 lane rides its 1x2 budget like the other time-series.
        TASK=T4  # queue letter stays T4C (log name); dispatch target is T4
        run prithvi_eo_v2_300_tl full_finetune 42
        run_anysat full_finetune 42 "${T4C_MICRO:-1}" "${T4C_ACC:-2}"
        ;;
    T5C)
        # Two-stage AnySat T5, separately reported under B2S. Defaults are
        # 125 epochs, freeze adapter after 62, micro-batch 1 x accumulation 2.
        # The measured short-run extrapolation is ~22h/seed vs ~36h online;
        # use the fast-disk RS_EVAL_CACHE_ROOT for the ~428GB dense cache.
        # Existing online/Colab runs keep their original protocol identity.
        TASK=T5  # queue letter stays T5C (log name); dispatch target is T5
        export ANYSAT_T5_STAGE=two
        export ANYSAT_FROZEN_CACHE=1
        # RS_EVAL_T5C_SEEDS narrows the chain to the seeds still missing, so a
        # top-up does not report the already-completed seeds as SKIP failures.
        for seed in ${RS_EVAL_T5C_SEEDS:-17 42 73}; do
            run_anysat frozen_adapter_two_stage "$seed"
        done
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
