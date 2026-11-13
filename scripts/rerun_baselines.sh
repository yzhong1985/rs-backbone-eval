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
#
# Never edit this file while a detached lane is running: bash reads the script
# lazily, so an insertion shifts its read offset and the chain dies with a
# bogus syntax error at whatever line the old offset now lands on (T11O20,
# 2026-11-08 — both runs had already finished, so no data was lost). Commit
# script changes before launching, and leave the tree alone until the lane ends.
set -uo pipefail
cd "$(dirname "$0")/.."

# The project's runtime paths and checkpoints live in .env; uv does not load it
# automatically, so source it explicitly. An explicitly exported
# RS_EVAL_RUNS_ROOT wins over .env so the completed-run guard below can be
# pointed at an isolated directory (tests), while production keeps using .env.
_runs_root_override="${RS_EVAL_RUNS_ROOT:-}"
set -a
# shellcheck disable=SC1091
# RS_EVAL_SKIP_ENV=1 documents that the caller (lane tests) supplies all
# runtime roots: sourcing ./.env would otherwise be able to override
# RS_EVAL_RERUN_LOGDIR-style values the test pinned (round-9 review note).
if [ "${RS_EVAL_SKIP_ENV:-0}" != "1" ]; then
    . ./.env
fi
set +a
if [ -n "$_runs_root_override" ]; then
    export RS_EVAL_RUNS_ROOT="$_runs_root_override"
fi
# Bring uv in from the standard location only when it is not already
# resolvable: callers (e.g. the lane tests' stub environment) must stay able
# to put a stub/uv first in PATH, and prepending unconditionally would
# override it and dispatch REAL runs under a mocked GPU (round-9 review).
if ! command -v uv > /dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
fi
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
# CPU-only lanes skip the GPU idle wait: no card is needed and they are meant
# to run beside a busy GPU lane. Gated here, BEFORE the wait (review round-9:
# the T11BA branch exported the flag after the check had already run, so the
# lane queued behind T9/T10 and exited 1 on hosts without a GPU).
CPU_LANES=" T11BA T11BA20 T11BB T11O T11O128 T11O12820 T11O20 "
if [ "${RS_EVAL_RERUN_CPU_ONLY:-0}" != "1" ] && [ "$CPU_LANES" != "${CPU_LANES/" $TASK "/}" ]; then
    RS_EVAL_RERUN_CPU_ONLY=1
fi
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
if [ "${RS_EVAL_RERUN_CPU_ONLY:-0}" != "1" ]; then
    # GPU lanes wait for an idle card; "no GPU" is a fatal misconfiguration.
    # CPU-only lanes (T11BA) skip both: no card needed, and running beside a
    # busy GPU lane is exactly the point (review 2026-11-03, round 7).
    # A WSL host keeps ~3.5 GB used by the Windows desktop, so the threshold is
    # configurable; raising it is a deployment decision, not a silent bypass
    # (review 2026-11-05, round 2).
    idle_limit="${RS_EVAL_GPU_IDLE_MIB:-2000}"
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -z "$used" ]; then echo "$(date -Is) no GPU visible"; exit 1; fi
        if [ "$used" -le "$idle_limit" ]; then break; fi
        echo "$(date -Is) GPU busy (${used} MiB, idle limit ${idle_limit}); waiting"
        sleep 300
    done
    echo "$(date -Is) GPU free; starting"
else
    echo "$(date -Is) CPU-only lane; skipping the GPU idle wait"
fi

failed=0
succeeded=0

stop_on_fail() {
    # A cell lane whose first seed hit a known bug must not keep burning the GPU
    # on the remaining seeds (2026-10-30: after seed 17 OOM'd, the chain started
    # seed 42 on the same broken code). Lanes that chain independent cells keep
    # the historical continue-on-error behaviour.
    if [ "${RS_EVAL_RERUN_STOP_ON_FAIL:-0}" = 1 ]; then
        echo "$(date -Is) STOP chain after $1 (RS_EVAL_RERUN_STOP_ON_FAIL=1)"
        exit 1
    fi
}

require_free_gb() {  # path need_gb label
    # A full disk must stop the lane before the first seed, not after a
    # two-hour run whose checkpoint write fails. The worker also leaves the
    # completed metric intact if the write fails; this is the earlier gate.
    local path=$1 need=$2 label=${3:-lane}
    local available
    available=$(df -Pk "$path" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}')
    if [ -z "$available" ]; then
        echo "$(date -Is) cannot read free space for $path" >&2
        exit 1
    fi
    if [ "$available" -lt "$need" ]; then
        echo "$(date -Is) $label needs ${need} GB free, only ${available} GB at $path" >&2
        exit 1
    fi
    echo "$(date -Is) disk preflight $label: ${available} GB free at $path"
}

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
        echo "$(date -Is) OK $rid $(grep -oE 'water_iou=[0-9.]+|mean_iou=[0-9.]+|test mIoU [0-9.]+' "$rlog" | tail -1)"
    else
        failed=$((failed + 1))
        echo "$(date -Is) FAIL $rid rc=$? (see $rlog)"
        stop_on_fail "$rid"
    fi
}

run_anysat() {  # protocol seed [micro_batch] [grad_accum]
    # AnySat's T1 adapter path OOMs at the official batch of 2; a micro-batch of
    # 1 with two accumulation steps keeps the *effective* batch (and fairness)
    # unchanged. Callers can override for cells where the full batch fits.
    export ANYSAT_WORKER_BATCH_SIZE="${3:-1}"
    export ANYSAT_WORKER_GRAD_ACCUM="${4:-2}"
    # B1 unfreezes the backbone: 256 crops exceed 24 GB even at micro-batch 1
    # (d5fdf40), so full fine-tuning keeps the documented 224 crop. B2 stays at
    # the shared 256 and the chosen crop is recorded in the run tags.
    if [ "$1" = "full_finetune" ]; then
        export ANYSAT_WORKER_TRAIN_CROP=224
    fi
    run anysat_base "$1" "$2"
    unset ANYSAT_WORKER_BATCH_SIZE ANYSAT_WORKER_GRAD_ACCUM
    unset ANYSAT_WORKER_TRAIN_CROP
}

run_named() {  # model protocol seed run_id
    # Same bookkeeping as run(), for cells whose id must carry a condition
    # (e.g. the T1 learning-rate grid) instead of the shared suffix.
    local model=$1 proto=$2 seed=$3 rid=$4
    local rlog="$LOGDIR/${rid}.log"
    local runs_root="${RS_EVAL_RUNS_ROOT:-/geodata/rs-eval/runs}"
    if [ -f "$runs_root/$rid/run.json" ] && \
        grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$runs_root/$rid/run.json" 2>/dev/null; then
        echo "$(date -Is) SKIP $rid: a completed run exists"
        failed=$((failed + 1))
        return
    fi
    echo "$(date -Is) START $rid"
    if uv run rs-eval dispatch --action train --task "$TASK" --model "$model" \
        --protocol "$proto" --seed "$seed" --run-id "$rid" > "$rlog" 2>&1; then
        succeeded=$((succeeded + 1))
        echo "$(date -Is) OK $rid $(grep -oE 'water_iou=[0-9.]+|mean_iou=[0-9.]+|test mIoU [0-9.]+' "$rlog" | tail -1)"
    else
        failed=$((failed + 1))
        echo "$(date -Is) FAIL $rid rc=$? (see $rlog)"
        stop_on_fail "$rid"
    fi
}

clay_t1_lr_grid() {
    # The 2026-10-21 audit showed the earlier Clay T1 grid only produced
    # head-only runs, so the 3e-4 used by T1FIX is a candidate, not a searched
    # value. Re-run the same three points on the full fine-tune path (seed 42,
    # selection on validation) so the T1 B1 cell has a real basis.
    local suffix="${RS_EVAL_RERUN_SUFFIX:-v3}"
    local task_before="$TASK"
    TASK=T1
    export CLAY_WORKER_BATCH_SIZE="$B1_EFFECTIVE_BATCH"
    for lr in 1e-4 3e-4 3e-5; do
        local tag
        tag=$(printf '%s' "$lr" | tr '.' 'p')
        export CLAY_WORKER_LR="$lr"
        run_named clay_v1_5 full_finetune 42 "clay_v1_5-T1-full_finetune-42-lr${tag}-${suffix}"
    done
    unset CLAY_WORKER_LR CLAY_WORKER_BATCH_SIZE
    TASK="$task_before"
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
        echo "$(date -Is) OK $rid $(grep -oE 'water_iou=[0-9.]+|mean_iou=[0-9.]+|test mIoU [0-9.]+' "$rlog" | tail -1)"
    else
        failed=$((failed + 1))
        echo "$(date -Is) FAIL $rid rc=$? (see $rlog)"
        stop_on_fail "$rid"
    fi
}

# B1 learning rates: one value per backbone, selected by the T1 log-grid search
# (anysat 1e-4, clay 3e-4, prithvi 3e-5) and held fixed across tasks. Declared
# here before any T2/T3/T5 B1 result existed; the search was never re-run
# against a test split.
B1_LR_ANYSAT=1e-4
B1_LR_CLAY=3e-4
B1_LR_PRITHVI=3e-5

# Effective batch for every B1 cell: AnySat's adapter path OOMs above a
# micro-batch of 1, so it runs 1 x 2 accumulation. The other two backbones use a
# real micro-batch of 2 so all three optimise on the same effective batch (the
# design doc allows a different micro-batch only if accumulation keeps the
# effective batch identical).
B1_EFFECTIVE_BATCH=2

b1_cell() {  # seeds... -- three backbones at their declared B1 lr
    local seeds="$*"
    export ANYSAT_WORKER_LR="$B1_LR_ANYSAT"
    for seed in $seeds; do run_anysat full_finetune "$seed"; done
    unset ANYSAT_WORKER_LR
    export CLAY_WORKER_LR="$B1_LR_CLAY"
    export CLAY_WORKER_BATCH_SIZE="$B1_EFFECTIVE_BATCH"
    for seed in $seeds; do run clay_v1_5 full_finetune "$seed"; done
    unset CLAY_WORKER_LR CLAY_WORKER_BATCH_SIZE
    export PRITHVI_WORKER_LR="$B1_LR_PRITHVI"
    export PRITHVI_WORKER_BATCH_SIZE="$B1_EFFECTIVE_BATCH"
    for seed in $seeds; do run prithvi_eo_v2_300_tl full_finetune "$seed"; done
    unset PRITHVI_WORKER_LR PRITHVI_WORKER_BATCH_SIZE
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
        # AnySat T3 full-finetune. The original cell budget was one seed (42);
        # a host with enough memory can run the full 17/42/73 set via
        # RS_EVAL_T3C_SEEDS. The 24 GB OOM was at 23.35 GiB, so a 32 GB host
        # should clear it without changing the recipe.
        TASK=T3  # queue letter stays T3C (log name); dispatch target is T3
        for seed in ${RS_EVAL_T3C_SEEDS:-42}; do
            run_anysat full_finetune "$seed" "${T3C_MICRO:-1}" "${T3C_ACC:-2}"
        done
        ;;
    T4C)
        # T4 AnySat full-finetune (landslide4sense). The original cell budget
        # is one seed (42); RS_EVAL_T4C_SEEDS can extend it. Prithvi is already
        # complete on the 4090 under ids ending -v1, so a 5090 rerun with a
        # different suffix would NOT skip it and would mix two GPU classes in
        # one cell. Prithvi is therefore opt-in only (RS_EVAL_T4C_INCLUDE_PRITHVI=1)
        # and is off by default; the 5090 host runs AnySat only.
        TASK=T4  # queue letter stays T4C (log name); dispatch target is T4
        for seed in ${RS_EVAL_T4C_SEEDS:-42}; do
            if [ "${RS_EVAL_T4C_INCLUDE_PRITHVI:-0}" = 1 ]; then
                run prithvi_eo_v2_300_tl full_finetune "$seed"
            fi
            run_anysat full_finetune "$seed" "${T4C_MICRO:-1}" "${T4C_ACC:-2}"
        done
        ;;
    T4B1)
        # T4 B1 on the 24 GB 4090: Clay then Prithvi, batch 2, 50 epochs.
        # Clay seed 17 is already running as ...-v1; this lane picks up the
        # remaining Clay seeds and all three Prithvi seeds. run_named keeps the
        # v1 ids consistent with that first run and does not overwrite a
        # completed cell.
        TASK=T4
        export CLAY_WORKER_LR="$B1_LR_CLAY"
        export CLAY_WORKER_BATCH_SIZE=2
        for seed in 42 73; do
            run_named clay_v1_5 full_finetune "$seed" \
                "clay_v1_5-T4-full_finetune-${seed}-v1"
        done
        unset CLAY_WORKER_LR CLAY_WORKER_BATCH_SIZE
        export PRITHVI_WORKER_LR="$B1_LR_PRITHVI"
        export PRITHVI_WORKER_BATCH_SIZE=2
        for seed in 17 42 73; do
            run_named prithvi_eo_v2_300_tl full_finetune "$seed" \
                "prithvi_eo_v2_300_tl-T4-full_finetune-${seed}-v1"
        done
        unset PRITHVI_WORKER_LR PRITHVI_WORKER_BATCH_SIZE
        ;;
    T4B1P)
        # Prithvi half of T4 B1, so the idle 20 GB host can fill a *new* cell
        # while the 4090 keeps Clay (whose seed 17 already lives there: a cell's
        # seeds never straddle hosts). T4 is single-date, six-band, 224 px, so
        # Prithvi B1 peaks far below 20 GB -- measured 8.5 GB on T5's 125-epoch
        # run. Ids match the T4B1 lane, and run_named skips completed cells, so
        # running the full T4B1 lane later only picks up what is still missing.
        TASK=T4
        export PRITHVI_WORKER_LR="$B1_LR_PRITHVI"
        export PRITHVI_WORKER_BATCH_SIZE=2
        for seed in 17 42 73; do
            run_named prithvi_eo_v2_300_tl full_finetune "$seed" \
                "prithvi_eo_v2_300_tl-T4-full_finetune-${seed}-v1"
        done
        unset PRITHVI_WORKER_LR PRITHVI_WORKER_BATCH_SIZE
        ;;
    T5B1C)
        # Clay T5 B1 (full fine-tune): the cell only has Prithvi today, so the
        # cross-model comparison at T5 has no second finetune to compare with.
        TASK=T5
        export CLAY_WORKER_LR="$B1_LR_CLAY"
        export CLAY_WORKER_BATCH_SIZE=2
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        for seed in 17 42 73; do
            run_named clay_v1_5 full_finetune "$seed" \
                "clay_v1_5-T5-full_finetune-${seed}-v1"
        done
        unset CLAY_WORKER_LR CLAY_WORKER_BATCH_SIZE
        ;;
    T4B1C)
        # Clay half of T4 B1 on the 24 GB host. Seed 17 already lives there
        # (...-v1); this lane adds 42 and 73 so the cell reaches three seeds
        # without touching Prithvi's cell, which runs on the 20 GB host.
        TASK=T4
        export CLAY_WORKER_LR="$B1_LR_CLAY"
        export CLAY_WORKER_BATCH_SIZE=2
        for seed in 42 73; do
            run_named clay_v1_5 full_finetune "$seed" \
                "clay_v1_5-T4-full_finetune-${seed}-v1"
        done
        unset CLAY_WORKER_LR CLAY_WORKER_BATCH_SIZE
        ;;
    T4DEM)
        # DIAGNOSTIC, NOT A RESULT. Landslide4Sense ships 14 channels (12 S2 +
        # slope + DEM); the benchmark protocol keeps only the six optical bands.
        # Three conditions, frozen probe, three seeds:
        #   opt - the six official bands (reproduces the formal cell)
        #   dem - six + DEM
        #   ds  - six + DEM + slope
        # Must run with RS_EVAL_RUNS_ROOT pointing at a diagnostics root: these
        # runs share task/model/protocol/seed with the T4 baseline, so writing
        # them into the main ledger would let canonical_runs pick one of them
        # instead of a baseline seed (review 2026-10-27, P1).
        TASK=T4
        # A crashed attempt leaves its run dir behind and dispatch refuses to
        # reuse it, so the suffix prefix is overridable for a clean retry.
        DEM_SUFFIX="${RS_EVAL_T4DEM_SUFFIX:-t4dem}"
        for condition in opt dem ds; do
            case "$condition" in
                opt) export CLAY_WORKER_T4_BANDS="BLUE,GREEN,RED,NIR_BROAD,SWIR_1,SWIR_2" ;;
                dem) export CLAY_WORKER_T4_BANDS="BLUE,GREEN,RED,NIR_BROAD,SWIR_1,SWIR_2,DEM" ;;
                ds) export CLAY_WORKER_T4_BANDS="BLUE,GREEN,RED,NIR_BROAD,SWIR_1,SWIR_2,DEM,SLOPE" ;;
            esac
            export RS_EVAL_RERUN_SUFFIX="${DEM_SUFFIX}-$condition"
            for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        done
        unset CLAY_WORKER_T4_BANDS RS_EVAL_RERUN_SUFFIX DEM_SUFFIX
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
    T1S)
        # T1S (Sentinel-1 VV/VH) after f69e703, with the same settings as the T1
        # lanes so optical and radar pair up: B2 at the worker defaults, then B1
        # at each backbone's T1 lr and effective batch 2 (AnySat 1x2, which
        # peaks at ~20 GB). Prithvi/OlmoEarth have no T1S path.
        # Every run also scores the Bolivia holdout (c13d9c2) as an auxiliary
        # metric; the Clay B2 v1 seeds predate that and are re-run as v2.
        TASK=T1S
        export RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout
        export RS_EVAL_RERUN_SUFFIX=v2
        for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        export RS_EVAL_RERUN_SUFFIX=v1
        for seed in 17 42 73; do run_anysat frozen_probe "$seed"; done
        export CLAY_WORKER_LR="$B1_LR_CLAY"
        export CLAY_WORKER_BATCH_SIZE="$B1_EFFECTIVE_BATCH"
        for seed in 17 42 73; do run clay_v1_5 full_finetune "$seed"; done
        unset CLAY_WORKER_LR CLAY_WORKER_BATCH_SIZE
        export ANYSAT_WORKER_LR="$B1_LR_ANYSAT"
        for seed in 17 42 73; do run_anysat full_finetune "$seed"; done
        unset ANYSAT_WORKER_LR RS_EVAL_RERUN_SUFFIX RS_EVAL_EXTRA_EVAL_SPLITS
        ;;
    T1SFIX)
        # The six T1S runs a concurrent metadata smoke (2026-10-26 16:05-16:13)
        # pushed into OOM; their v1 records are quarantined as
        # *.oom-concurrent-smoke. Same settings as the T1S lane, suffix v2.
        TASK=T1S
        export RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout
        export RS_EVAL_RERUN_SUFFIX=v2
        for seed in 42 73; do run_anysat frozen_probe "$seed"; done
        export CLAY_WORKER_LR="$B1_LR_CLAY"
        export CLAY_WORKER_BATCH_SIZE="$B1_EFFECTIVE_BATCH"
        for seed in 17 42 73; do run clay_v1_5 full_finetune "$seed"; done
        unset CLAY_WORKER_LR CLAY_WORKER_BATCH_SIZE
        export ANYSAT_WORKER_LR="$B1_LR_ANYSAT"
        run_anysat full_finetune 17
        unset ANYSAT_WORKER_LR RS_EVAL_RERUN_SUFFIX RS_EVAL_EXTRA_EVAL_SPLITS
        ;;
    T1HOLD)
        # T1 re-run so every formal cell also carries the Bolivia holdout
        # (26f0f3e) and, on the frozen track, the D2 pixel and metadata sweeps. Same
        # settings as the T1/T1FIX lanes; the new runs become the canonical
        # seeds of their cells. AnySat (hours per seed) is lane T1HOLDA.
        TASK=T1
        export RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout
        export RS_EVAL_RERUN_SUFFIX=hold-v1
        export RS_EVAL_PERTURB="default;metadata"
        for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl frozen_probe "$seed"; done
        for seed in 17 42 73; do run olmoearth_v1_2_tiny frozen_adapter "$seed"; done
        for seed in 17 42 73; do run olmoearth_v1_2_base frozen_adapter "$seed"; done
        unset RS_EVAL_PERTURB
        export CLAY_WORKER_LR="$B1_LR_CLAY"
        export CLAY_WORKER_BATCH_SIZE="$B1_EFFECTIVE_BATCH"
        for seed in 17 42 73; do run clay_v1_5 full_finetune "$seed"; done
        unset CLAY_WORKER_LR CLAY_WORKER_BATCH_SIZE
        export PRITHVI_WORKER_LR="$B1_LR_PRITHVI"
        export PRITHVI_WORKER_BATCH_SIZE="$B1_EFFECTIVE_BATCH"
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl full_finetune "$seed"; done
        unset PRITHVI_WORKER_LR PRITHVI_WORKER_BATCH_SIZE
        unset RS_EVAL_RERUN_SUFFIX RS_EVAL_EXTRA_EVAL_SPLITS
        ;;
    T1HOLDA)
        # AnySat part of T1HOLD (hours per seed), split out so it can queue
        # behind other work. Single-host lane; the _B2/_B1 halves below run the
        # same cells on two machines, where the 20 GB card cannot host B1.
        TASK=T1
        export RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout
        export RS_EVAL_RERUN_SUFFIX=hold-v1
        export RS_EVAL_PERTURB="default;metadata"
        for seed in 17 42 73; do run_anysat frozen_adapter "$seed"; done
        unset RS_EVAL_PERTURB
        export ANYSAT_WORKER_LR="$B1_LR_ANYSAT"
        for seed in 17 42 73; do run_anysat full_finetune "$seed"; done
        unset ANYSAT_WORKER_LR RS_EVAL_RERUN_SUFFIX RS_EVAL_EXTRA_EVAL_SPLITS
        ;;
    T1HOLDA_B2)
        # Frozen half of T1HOLDA: clean + four pixel perturbations + the date
        # sweep + Bolivia, three seeds. Fits a 20 GB card (~12-17 GB).
        TASK=T1
        export RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout
        export RS_EVAL_RERUN_SUFFIX=hold-v1
        export RS_EVAL_PERTURB="default;metadata"
        for seed in 17 42 73; do run_anysat frozen_adapter "$seed"; done
        unset RS_EVAL_PERTURB RS_EVAL_RERUN_SUFFIX RS_EVAL_EXTRA_EVAL_SPLITS
        ;;
    T1HOLDA_B1)
        # Full fine-tune half of T1HOLDA. Needs the 24 GB host: at the
        # documented 224 crop it peaks near 20.4 GB, above a 20 GB card. The
        # seeds stay together on one host (AGENTS.md).
        TASK=T1
        export RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout
        export RS_EVAL_RERUN_SUFFIX=hold-v1
        export ANYSAT_WORKER_LR="$B1_LR_ANYSAT"
        for seed in 17 42 73; do run_anysat full_finetune "$seed"; done
        unset ANYSAT_WORKER_LR RS_EVAL_RERUN_SUFFIX RS_EVAL_EXTRA_EVAL_SPLITS
        ;;
    T1S_WL)
        # DIAGNOSTIC, NOT A RESULT. Is Clay using the wavelength numbers or only
        # the radar convention? Four settings, frozen probe, three seeds:
        #   base - Clay's own sentinel-1-rtc placeholders (vv 3.5, vh 4.0 um)
        #   phys - physical C-band (~5.5e4 um), far outside the pretraining range
        #   swap - 4.0/3.5, does the value only identify the polarisation?
        #   same - both bands identical, removes that identification entirely
        # It must run with RS_EVAL_RUNS_ROOT pointing at a diagnostics root: the
        # four conditions share task/model/protocol/seed with the T1S baseline,
        # so writing them into the main ledger would let canonical_runs pick an
        # ablation run in place of a baseline seed (review 2026-10-27, P1).
        TASK=T1S
        for condition in base phys swap same; do
            case "$condition" in
                base) unset CLAY_WORKER_WAVELENGTHS_UM ;;
                phys) export CLAY_WORKER_WAVELENGTHS_UM="55000,55000" ;;
                swap) export CLAY_WORKER_WAVELENGTHS_UM="4.0,3.5" ;;
                same) export CLAY_WORKER_WAVELENGTHS_UM="3.5,3.5" ;;
            esac
            export RS_EVAL_RERUN_SUFFIX="wl-$condition"
            for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        done
        unset RS_EVAL_RERUN_SUFFIX CLAY_WORKER_WAVELENGTHS_UM
        ;;
    T8B2)
        # T8 (GF-2 four-band) frozen track on the chip protocol (2cdf9dd) and the
        # footprint-grouped manifest. Clay reads the four bands natively;
        # Prithvi/AnySat train a band adapter (frozen_adapter). AnySat needs 1x2.
        TASK=T8
        export RS_EVAL_MANIFEST="${RS_EVAL_T8_MANIFEST:-$RS_EVAL_DATA_ROOT/manifests/T8/56bfce995f768c80ab600bf3bff95b9b48cd6f650baf1674d7e8235a154b4c85.jsonl}"
        export RS_EVAL_RERUN_SUFFIX=v1
        for seed in 17 42 73; do run clay_v1_5 frozen_probe "$seed"; done
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl frozen_adapter "$seed"; done
        for seed in 17 42 73; do run_anysat frozen_adapter "$seed"; done
        unset RS_EVAL_MANIFEST RS_EVAL_RERUN_SUFFIX
        ;;
    T1FIX)
        # The T1 B1 cells still mix pre-fix runs: AnySat seed 42 and all three
        # Prithvi seeds predate ffdc6db/7377fb6, and Clay's 3e-4 runs only ever
        # trained the head (tagged superseded). Re-run the missing seeds at the
        # lr the T1 log-grid search selected for each backbone.
        TASK=T1
        export ANYSAT_WORKER_LR="$B1_LR_ANYSAT"
        run_anysat full_finetune 42
        unset ANYSAT_WORKER_LR
        export CLAY_WORKER_LR="$B1_LR_CLAY"
        export CLAY_WORKER_BATCH_SIZE="$B1_EFFECTIVE_BATCH"
        for seed in 17 42 73; do run clay_v1_5 full_finetune "$seed"; done
        unset CLAY_WORKER_LR CLAY_WORKER_BATCH_SIZE
        export PRITHVI_WORKER_LR="$B1_LR_PRITHVI"
        export PRITHVI_WORKER_BATCH_SIZE="$B1_EFFECTIVE_BATCH"
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl full_finetune "$seed"; done
        unset PRITHVI_WORKER_LR PRITHVI_WORKER_BATCH_SIZE
        ;;
    T2B1)
        TASK=T2  # queue letter stays T2B1 (log name)
        b1_cell 17 42 73
        ;;
    T3B1)
        # T3 keeps the repo budget (50 epochs): the paper's 150-epoch anchor
        # assumes its UPerNet decoder and 80/20 split, and T3's test set is only
        # 28 chips, so 150 epochs would spend ~63h on a metric that cannot
        # resolve the difference. Labelled as a non-paper budget in the report.
        TASK=T3
        clay_t1_lr_grid
        b1_cell 17 42 73
        ;;
    T3B0EPOCH)
        # Diagnostic only, and a *hypothesis test* rather than a finding: the
        # T3 gap against the paper (mIoU 0.156 vs 0.294) has not been attributed
        # to any single cause. Paper v2 (2412.02732v2) trains burn intensity for
        # 150 epochs; the repo budget is 50. This lane reruns B0 at 100 and 150
        # to see how much of the gap a longer schedule explains -- if it does
        # not, the loss/recipe/decoder also differ and must be investigated.
        # The worker marks any non-official epoch count as a smoke budget, so
        # these never enter the formal matrix: read them from the ledger and
        # quote them as a protocol probe, not as results.
        TASK=T3
        for epochs in 100 150; do
            export PRITHVI_WORKER_EPOCHS="$epochs"
            export RS_EVAL_RERUN_SUFFIX="e$epochs"
            for seed in 17 42 73; do
                run prithvi_eo_v2_300_tl prithvi_paper_protocol "$seed"
            done
        done
        unset PRITHVI_WORKER_EPOCHS RS_EVAL_RERUN_SUFFIX
        ;;
    T5B1)
        # Only Prithvi fits the 4090 for a 125-epoch, three-date full fine-tune:
        # AnySat's frozen path already peaks at 17.7GB (OOM risk with gradients)
        # and Clay needs roughly 50h/seed. Reported as a partial B1 cell.
        TASK=T5
        export PRITHVI_WORKER_LR="$B1_LR_PRITHVI"
        for seed in 17 42 73; do run prithvi_eo_v2_300_tl full_finetune "$seed"; done
        unset PRITHVI_WORKER_LR
        ;;
    T10B2)
        # FAIR1M rotated detection, frozen track (B2D), LIM. The budget is the
        # registered 25,000 optimizer steps (50k tiles at micro 1 x accum 2),
        # which is ~3.1 h per seed at the measured 0.447 s/step. LIM_WORKER
        # _EPOCHS=12 is only the loop bound; the step budget stops the run first.
        # v8 is the 2026-11-03 normalization variant (LIM ImageNet statistics,
        # preprocessing_id fair1m_rgb_imagenet). It is owned by the RTX 5090
        # night node: seeds 17/42/73 all stay on that host. Do not start v8 on
        # the 4090 -- two ledgers cannot see each other and would both write
        # lim_mit_huge-T10-rotated_frozen-train-17-v8 (review 2026-11-05).
        # RS_EVAL_T10B2_SUFFIX is required and is part of the run id, mirroring
        # the T11 -4090 suffix; a host change also changes the cell.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1  # a bug in seed 1 must not burn the next seeds
        TASK=T10
        : "${RS_EVAL_MANIFEST:?pin the T10 manifest, e.g. .../manifests/T10/holdout_1024_s824_seed17.jsonl}"
        : "${RS_EVAL_T10B2_SUFFIX:?set the host suffix (5090 for v8; a rerun on another host is a new cell)}"
        # Optional wall-clock gate: do not start a seed after HH:MM. The cutoff
        # is anchored to the same night: a lane started in the evening uses the
        # next day's HH:MM, a lane started after midnight uses today's. Comparing
        # raw HHMM would silently skip everything before midnight (round 12).
        # Only use early-morning cutoffs (< 12:00); a 14:00 start with an 18:00
        # cutoff would be interpreted as tomorrow evening.
        start_before="${RS_EVAL_RERUN_START_BEFORE:-}"
        if [ -n "$start_before" ]; then
            now_s=$(date +%s)
            now_hm=$(date +%H%M)
            now_h=$((10#${now_hm:0:2}))
            now_m=$((10#${now_hm:2:2}))
            midnight=$((now_s - now_h * 3600 - now_m * 60))
            cut_h=$((10#${start_before%%:*}))
            cut_m=$((10#${start_before##*:}))
            cutoff=$((midnight + cut_h * 3600 + cut_m * 60))
            if [ "$now_h" -ge 12 ]; then
                cutoff=$((cutoff + 86400))
            fi
        fi
        export LIM_WORKER_TOTAL_STEPS=25000
        export LIM_WORKER_EPOCHS=12
        # v1 OOM'd on the full anchor x GT IoU matrix (fixed by chunking);
        # v2/v3 diverged with loss_rpn_reg=inf in epoch 2 — 115 zero-area
        # annotations, one of which reaches a training tile and makes
        # log(w/anchor) -inf. v4 drops min-side < 0.5 px boxes from targets and
        # ground truth and caps the log floor; v3 already added grad clipping and
        # fail-fast. v5 ran the full budget but scored mAP 0.0000 because
        # `_roi_features` returned rows grouped by FPN level (fixed in 4b8b197,
        # registered as T10_ROI_ORDER). v6 is the first post-fix run but still
        # predates T10_DETECTOR / T10_RPN_NMS, so its scores are stale. v7 is the
        # first run on the fixed detector: seed 17 reached mAP 0.0565 at the full
        # 25k budget (2026-11-02). RS_EVAL_T10B2_SEEDS narrows the chain to the
        # seeds still missing, so a top-up does not report the already-completed
        # seed 17 as a SKIP failure. v8 is the first T10 run under the
        # 2026-11-03 normalization decision (LIM uses its ImageNet pretraining
        # statistics, fixed across seeds; preprocessing_id fair1m_rgb_imagenet).
        # v7 used per-seed train-tile statistics, is a different variant, and
        # stays in the ledger.
        # Pin RS_EVAL_MANIFEST: a rebuilt manifest changes its sha256 and the
        # seeds of one cell must share it.
        for seed in ${RS_EVAL_T10B2_SEEDS:-17 42 73}; do
            if [ -n "$start_before" ] && [ "$(date +%s)" -ge "$cutoff" ]; then
                echo "$(date -Is) seed ${seed} skipped: past RS_EVAL_RERUN_START_BEFORE=${start_before}"
                break
            fi
            run_named lim_mit_huge rotated_frozen "$seed" \
                "lim_mit_huge-T10-rotated_frozen-train-${seed}-v8-${RS_EVAL_T10B2_SUFFIX}"
        done
        unset LIM_WORKER_TOTAL_STEPS LIM_WORKER_EPOCHS
        ;;
    T9B2P)
        # iSAID B2U, Prithvi, all three seeds on the 20 GB host (one host per
        # cell). v4 is the first rerun after 0e88549 (padding) and the
        # 2026-11-03 normalization decision: dataset statistics are now sampled
        # once with a fixed seed-0, so the old unsuffixed runs are audit
        # excluded (NORM_DECISION) and this cell must run the full seed set.
        # Budget is 25,000 optimizer steps = 200k tiles at effective batch 8;
        # the worker breaks on steps, so the 50-epoch default is only the loop
        # bound. RS_EVAL_MANIFEST must be pinned by the caller: the most
        # recently written manifest is not necessarily this cell's.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1  # a bug in seed 1 must not burn the next seeds
        TASK=T9
        : "${RS_EVAL_MANIFEST:?pin the T9 manifest, e.g. .../manifests/T9/holdout_1024_s512_seed17.jsonl}"
        export PRITHVI_WORKER_TOTAL_STEPS=25000
        for seed in 17 42 73; do
            run_named prithvi_eo_v2_300_tl upernet_frozen "$seed" \
                "prithvi_eo_v2_300_tl-T9-upernet_frozen-train-${seed}-v4"
        done
        unset PRITHVI_WORKER_TOTAL_STEPS
        ;;
    T9B2C)
        # iSAID B2U, Clay, all three seeds on the 20 GB host. v4 keeps the same
        # versioning cause as T9B2P: statistics sampling fixed at seed 0 plus
        # the padding change (review 2026-11-03: the Clay T9 cell had no rerun
        # channel at all). Manifest pinning is the same requirement as above.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        TASK=T9
        : "${RS_EVAL_MANIFEST:?pin the T9 manifest, e.g. .../manifests/T9/holdout_1024_s512_seed17.jsonl}"
        # Clay worker defaults to total_steps=0 (a full 50 epochs = ~1.19M tiles
        # vs the 200k protocol budget; review 2026-11-03 R4-1), so set the step
        # budget exactly like the other B2U lanes. Smoke the Clay T9 worker for
        # throughput/memory before launching the lane.
        export CLAY_WORKER_TOTAL_STEPS=25000
        for seed in 17 42 73; do
            run_named clay_v1_5 upernet_frozen "$seed" \
                "clay_v1_5-T9-upernet_frozen-train-${seed}-v4"
        done
        unset CLAY_WORKER_TOTAL_STEPS
        ;;
    T11BA)
        # T11 AEF track, both declared heads, all three seeds. CPU-bound:
        # safe to run beside a GPU lane, but cap the threads so the lanes'
        # dataloaders keep their cores (review 2026-11-03 R6: a handwritten
        # 6-run dispatch loop ran the first batch; any future rerun goes
        # through this lane with fresh -v ids, never the unsuffixed ones).
        # The CPU-only gating already happened at the top (CPU_LANES), so the
        # branch only carries the thread cap (review round-9 close-out).
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export OMP_NUM_THREADS=8
        TASK=T11
        suffix="${RS_EVAL_T11BA_SUFFIX:?first batch ran unsuffixed; use e.g. RS_EVAL_T11BA_SUFFIX=v2 for a rerun}"
        # Optional seed filter for a targeted rerun (e.g. only the two s42 runs
        # whose commit is `-dirty`). Default stays all three seeds.
        for seed in ${RS_EVAL_T11BA_SEEDS:-17 42 73}; do
            run_named aef_annual_2019 aef_probe_mlp "$seed" \
                "aef_annual_2019-T11-aef_probe_mlp-train-${seed}-${suffix}"
            run_named aef_annual_2019 aef_probe_upernet "$seed" \
                "aef_annual_2019-T11-aef_probe_upernet-train-${seed}-${suffix}"
        done
        unset OMP_NUM_THREADS
        ;;
    T11BA20)
        # T11 AEF budget-sensitivity diagnostic: 20 epochs, both heads. Seeds
        # default to 17; the 3-seed extension passes SEEDS="42 73". Separate protocols and tracks (BA20/BAM20) so the 8-epoch AEF
        # cells keep the pre-declared budget. Pairs with T11O20 so the
        # product gap can be separated from the budget effect; not a headline
        # cell and never a replacement for BA/BAM.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export OMP_NUM_THREADS=8
        TASK=T11
        suffix="${RS_EVAL_T11BA20_SUFFIX:-v1}"
        # Seed filter so the 3-seed extension does not re-run the seed-17 cell.
        for seed in ${RS_EVAL_T11BA20_SEEDS:-17}; do
            run_named aef_annual_2019 aef_probe_mlp_20ep "$seed" \
                "aef_annual_2019-T11-aef_probe_mlp_20ep-train-${seed}-${suffix}"
            run_named aef_annual_2019 aef_probe_upernet_20ep "$seed" \
                "aef_annual_2019-T11-aef_probe_upernet_20ep-train-${seed}-${suffix}"
        done
        unset OMP_NUM_THREADS
        ;;
    T11O128)
        # T11O128: full 128-dim OlmoEarth embedding, same probes, three seeds.
        # CPU-bound like T11O; separate task/track so the 64-dim cell stays.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export OMP_NUM_THREADS=8
        TASK=T11O128
        suffix="${RS_EVAL_T11O128_SUFFIX:-v1}"
        for seed in ${RS_EVAL_T11O128_SEEDS:-17 42 73}; do
            run_named olmoearth_v13_embedding olmoearth_v13_probe_mlp_128 "$seed" \
                "olmoearth_v13_embedding-T11O128-olmoearth_v13_probe_mlp_128-train-${seed}-${suffix}"
            run_named olmoearth_v13_embedding olmoearth_v13_probe_upernet_128 "$seed" \
                "olmoearth_v13_embedding-T11O128-olmoearth_v13_probe_upernet_128-train-${seed}-${suffix}"
        done
        unset OMP_NUM_THREADS
        ;;
    T11O12820)
        # T11O128 budget-sensitivity diagnostic: full 128 dims, 20 epochs.
        # Seeds default to 17; the 3-seed extension passes SEEDS="42 73".
        # Completes the product x budget grid; the 64-dim axis is T11O20. Separate tracks (BOE12820/BOEM12820), not a headline cell.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export OMP_NUM_THREADS=8
        TASK=T11O128
        suffix="${RS_EVAL_T11O12820_SUFFIX:-v1}"
        for seed in ${RS_EVAL_T11O12820_SEEDS:-17}; do
            run_named olmoearth_v13_embedding olmoearth_v13_probe_mlp_128_20ep "$seed" \
                "olmoearth_v13_embedding-T11O128-olmoearth_v13_probe_mlp_128_20ep-train-${seed}-${suffix}"
            run_named olmoearth_v13_embedding olmoearth_v13_probe_upernet_128_20ep "$seed" \
                "olmoearth_v13_embedding-T11O128-olmoearth_v13_probe_upernet_128_20ep-train-${seed}-${suffix}"
        done
        unset OMP_NUM_THREADS
        ;;
    T11O20)
        # T11O budget-sensitivity diagnostic: 20 epochs, both heads. Seeds
        # default to 17; the 3-seed extension passes SEEDS="42 73". Separate protocols and tracks (BOE20/BOEM20) so the 8-epoch
        # BOE/BOEM cells keep the pre-declared budget, same convention as
        # BBF20. Not a headline cell.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export OMP_NUM_THREADS=8
        TASK=T11O
        suffix="${RS_EVAL_T11O20_SUFFIX:-v1}"
        for seed in ${RS_EVAL_T11O20_SEEDS:-17}; do
            run_named olmoearth_v13_embedding olmoearth_v13_probe_mlp_20ep "$seed" \
                "olmoearth_v13_embedding-T11O-olmoearth_v13_probe_mlp_20ep-train-${seed}-${suffix}"
            run_named olmoearth_v13_embedding olmoearth_v13_probe_upernet_20ep "$seed" \
                "olmoearth_v13_embedding-T11O-olmoearth_v13_probe_upernet_20ep-train-${seed}-${suffix}"
        done
        unset OMP_NUM_THREADS
        ;;
    T11O)
        # T11O OlmoEarth v1.3 annual-embedding track (T11O), both declared
        # heads, three seeds. CPU-bound like T11BA: the heads are identical and
        # the cell must stay on CPU so it is comparable with the AEF numbers.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export OMP_NUM_THREADS=8
        TASK=T11O
        suffix="${RS_EVAL_T11O_SUFFIX:-v1}"
        for seed in ${RS_EVAL_T11O_SEEDS:-17 42 73}; do
            run_named olmoearth_v13_embedding olmoearth_v13_probe_mlp "$seed" \
                "olmoearth_v13_embedding-T11O-olmoearth_v13_probe_mlp-train-${seed}-${suffix}"
            run_named olmoearth_v13_embedding olmoearth_v13_probe_upernet "$seed" \
                "olmoearth_v13_embedding-T11O-olmoearth_v13_probe_upernet-train-${seed}-${suffix}"
        done
        unset OMP_NUM_THREADS
        ;;
    T11BB)
        # T11 BB track (LIM native input), single/median, three seeds, CPU.
        # Round 2 decision (reviewer 2026-11-04): the lane IS the CPU cell --
        # CUDA_VISIBLE_DEVICES is cleared so the device-aware worker cannot
        # grab the 4090 beside another lane; the 20-bin LTAE2d cell runs as
        # the separate -4090 cell (2026-11-04 decision; the Colab/A100 notebook
        # is now the fallback). Same discipline as T11BA: mandatory suffix,
        # thread cap, no handwritten dispatches.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export OMP_NUM_THREADS=8
        # CPU cell by contract (reviewer 2026-11-04): the device-aware worker
        # must see no CUDA here. Clearing the variable (empty string, not
        # just assuming the host is free) is deliberate; anyone deliberately
        # wanting a GPU run from this lane must edit the lane, not the env.
        export CUDA_VISIBLE_DEVICES=""
        TASK=T11
        suffix="${RS_EVAL_T11BB_SUFFIX:?no unsuffixed runs from this lane; use e.g. RS_EVAL_T11BB_SUFFIX=cpu}"
        for seed in 17 42 73; do
            run_named lim_mit_huge lim_s2_single "$seed" \
                "lim_mit_huge-T11-lim_s2_single-train-${seed}-${suffix}"
            run_named lim_mit_huge lim_s2_median "$seed" \
                "lim_mit_huge-T11-lim_s2_median-train-${seed}-${suffix}"
        done
        unset OMP_NUM_THREADS CUDA_VISIBLE_DEVICES
        ;;
    T11BB4090)
        # Formal T11 BB cell (2026-11-04 decision): the local 4090 is the
        # formal host, so the whole cell stays on one GPU class. LIM runs the
        # single and 20-bin protocols (median withdrawn); AnySat, Clay and
        # OlmoEarth Tiny add the same seeds under their single protocols.
        # First batch suffix 4090; reruns must use a new suffix (v2...). The
        # lane never clears CUDA_VISIBLE_DEVICES: it is a GPU lane by contract.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export OMP_NUM_THREADS=8
        TASK=T11
        suffix="${RS_EVAL_T11BB4090_SUFFIX:-4090}"
        for seed in 17 42 73; do
            run_named lim_mit_huge lim_s2_single "$seed" \
                "lim_mit_huge-T11-lim_s2_single-train-${seed}-${suffix}"
            run_named lim_mit_huge lim_s2_20bin "$seed" \
                "lim_mit_huge-T11-lim_s2_20bin-train-${seed}-${suffix}"
            run_named anysat_base anysat_s2_single "$seed" \
                "anysat_base-T11-anysat_s2_single-train-${seed}-${suffix}"
            run_named anysat_base anysat_s2_20bin "$seed" \
                "anysat_base-T11-anysat_s2_20bin-train-${seed}-${suffix}"
            run_named clay_v1_5 clay_s2_single "$seed" \
                "clay_v1_5-T11-clay_s2_single-train-${seed}-${suffix}"
            run_named clay_v1_5 clay_s2_20bin "$seed" \
                "clay_v1_5-T11-clay_s2_20bin-train-${seed}-${suffix}"
            run_named olmoearth_v1_2_tiny olmoearth_s2_single "$seed" \
                "olmoearth_v1_2_tiny-T11-olmoearth_s2_single-train-${seed}-${suffix}"
            run_named olmoearth_v1_2_tiny olmoearth_s2_20bin "$seed" \
                "olmoearth_v1_2_tiny-T11-olmoearth_s2_20bin-train-${seed}-${suffix}"
        done
        unset OMP_NUM_THREADS
        ;;
    T11BBF4090)
        # T11 LIM full fine-tune on the 20-bin + LTAE2d input (B1 track, BBF).
        # Keep it out of the BBT frozen table. One checkpoint over all 20 frames
        # is both fastest and still small: smoke peak 12.4 GB, one epoch 5.1 min
        # (8 frames: 12.0 GB / 6.8 min; 4: 11.2 GB / 8.3 min; 1: 10.6 GB / 21.9 min).
        require_free_gb "${RS_EVAL_RUNS_ROOT:-/geodata/rs-eval/runs}" "${RS_EVAL_MIN_FREE_GB:-20}" T11BBF4090
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export LIM_T11_BATCH=1
        export LIM_T11_ACCUM=8
        export LIM_T11_CHECKPOINT_FRAMES=20
        TASK=T11
        suffix="${RS_EVAL_T11BBF_SUFFIX:-4090}"
        for seed in 17 42 73; do
            run_named lim_mit_huge lim_s2_20bin_finetune "$seed" \
                "lim_mit_huge-T11-lim_s2_20bin_finetune-train-${seed}-${suffix}"
        done
        unset LIM_T11_BATCH LIM_T11_ACCUM LIM_T11_CHECKPOINT_FRAMES
        ;;
    T11BBFLR4090)
        # LR exploration for the T11 BBF full fine-tune. The baseline
        # lim_s2_20bin_finetune keeps head 1e-3 / backbone 1e-4; these two cells
        # lower only the backbone LR (5e-5 and 1e-5) with the same 8-epoch,
        # effective-batch-8 recipe. Each protocol is a distinct variant under
        # the BBF track and is ranked on validation, never on test.
        : "${RS_EVAL_MANIFEST:?pin the T11 manifest}"
        require_free_gb "${RS_EVAL_RUNS_ROOT:-/geodata/rs-eval/runs}" "${RS_EVAL_MIN_FREE_GB:-20}" T11BBFLR4090
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export LIM_T11_BATCH=1
        export LIM_T11_ACCUM=8
        export LIM_T11_CHECKPOINT_FRAMES=20
        TASK=T11
        suffix="${RS_EVAL_T11BBFLR_SUFFIX:-4090}"
        for proto in lim_s2_20bin_finetune_lr5e5 lim_s2_20bin_finetune_lr1e5; do
            for seed in 17 42 73; do
                run_named lim_mit_huge "$proto" "$seed" \
                    "lim_mit_huge-T11-${proto}-train-${seed}-${suffix}"
            done
        done
        unset LIM_T11_BATCH LIM_T11_ACCUM LIM_T11_CHECKPOINT_FRAMES
        ;;
    T11BBF20)
        # Budget-scaling BBF track: the same 20-bin LTAE2d full fine-tune as
        # T11BBF, but 20 epochs. The protocol declares epochs=20 and enters
        # BBF20, never the 8-epoch BBF cell. Pass the concrete protocol id whose
        # backbone LR matches the screening winner; never edit a protocol YAML
        # in place after it has runs.
        : "${RS_EVAL_MANIFEST:?pin the T11 manifest}"
        require_free_gb "${RS_EVAL_RUNS_ROOT:-/geodata/rs-eval/runs}" "${RS_EVAL_MIN_FREE_GB:-20}" T11BBF20
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        export LIM_T11_BATCH=1
        export LIM_T11_ACCUM=8
        export LIM_T11_CHECKPOINT_FRAMES=20
        TASK=T11
        # lr5e-5 is the pre-declared LR-screen winner. Only this 20ep protocol
        # is mapped to BBF20; an unselected LR must be a new protocol/track.
        proto="${RS_EVAL_T11BBF20_PROTOCOL:-lim_s2_20bin_finetune_20ep_lr5e5}"
        suffix="${RS_EVAL_T11BBF20_SUFFIX:-4090}"
        for seed in 17 42 73; do
            run_named lim_mit_huge "$proto" "$seed" \
                "lim_mit_huge-T11-${proto}-train-${seed}-${suffix}"
        done
        unset LIM_T11_BATCH LIM_T11_ACCUM LIM_T11_CHECKPOINT_FRAMES
        ;;
    T9B2)
        # iSAID B2U, LIM, all three seeds on the 4090 (the cell stays on one
        # host/GPU class). v4 is the first run after 0e88549 (padding) and the
        # 2026-11-03 normalization decision: LIM now uses its pretraining
        # ImageNet statistics, fixed across seeds, so the old v3 / -42 / -73-r2
        # records are superseded and must not be mixed into the cell.
        # Budget is the registered 25,000 optimizer steps = 200k tiles at
        # effective batch 8 (micro 1 x accum 8); the worker breaks on steps, so
        # the 50-epoch default is only the loop bound.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1  # a bug in seed 1 must not burn the next seeds
        TASK=T9
        : "${RS_EVAL_MANIFEST:?pin the T9 manifest, e.g. .../manifests/T9/holdout_1024_s512_seed17.jsonl}"
        export LIM_WORKER_TOTAL_STEPS=25000
        for seed in 17 42 73; do
            run_named lim_mit_huge upernet_frozen "$seed" \
                "lim_mit_huge-T9-upernet_frozen-train-${seed}-v4"
        done
        unset LIM_WORKER_TOTAL_STEPS
        ;;
    T9B2R)
        # Retry of iSAID LIM seed 73 after the 2026-10-31 run died on a transient
        # rasterio "libpng: IDAT: CRC error" (all 1,869 iSAID PNGs verify and
        # decode clean, so the file is not corrupt). New run id: the failed
        # record stays in the ledger as evidence.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        TASK=T9
        : "${RS_EVAL_MANIFEST:?pin the T9 manifest, e.g. .../manifests/T9/holdout_1024_s512_seed17.jsonl}"
        export LIM_WORKER_TOTAL_STEPS=25000
        run_named lim_mit_huge upernet_frozen 73 \
            "lim_mit_huge-T9-upernet_frozen-train-73-r2"
        unset LIM_WORKER_TOTAL_STEPS
        ;;
    ANYFIX)
        # The T1/T2 AnySat B1 runs were launched at the 256 default and OOM'd;
        # redo them at the documented 224 crop. The declared B1 lr must be
        # exported explicitly: the worker default is 1e-3, not the selected
        # 1e-4, and the first attempt silently fell back to it (review
        # 2026-10-22, second round).
        TASK=T1
        export ANYSAT_WORKER_LR="$B1_LR_ANYSAT"
        run_anysat full_finetune 42
        TASK=T2
        for seed in 17 42 73; do run_anysat full_finetune "$seed"; done
        unset ANYSAT_WORKER_LR
        ;;
    T11PANSAT)
        # AnySat as the third T11P student. Fixed T11P manifest; effective
        # batch 2 always comes from micro-batch 1 x accumulation 2 (the adapter
        # + frozen backbone path peaks near the 24 GB limit). Per seed the raw
        # probe runs first: it trains and publishes the shared 6->10 adapter,
        # which the other readouts load and freeze. The B stage uses the wide
        # head after the T11D ablation showed a stable +0.03 over the 128-d
        # projector. A failure in seed 17 stops the lane: the six paths already
        # passed CPU smoke, so a repeated failure is a real regression.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        TASK=T11P
        export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11P/3155a8c5e4dc5c005037b0b41e261530c3eb85d48358c118e67f689ff0d6fb96.jsonl
        export ANYSAT_T11P_BATCH=1
        export ANYSAT_T11P_ACCUM=2
        for seed in 17 42 73; do
            # The raw probe trains and publishes the shared 6->10 adapter, so
            # every later protocol for that seed resolves the same input path.
            run_named anysat_base anysat_t11p_probe_raw "$seed" \
                "anysat_base-T11P-anysat_t11p_probe_raw-train-${seed}-v3"
            run_named anysat_base anysat_t11p_distill_o128 "$seed" \
                "anysat_base-T11P-anysat_t11p_distill_o128-train-${seed}-v3"
            run_named anysat_base anysat_t11p_probe_s128 "$seed" \
                "anysat_base-T11P-anysat_t11p_probe_s128-train-${seed}-v3"
            run_named anysat_base anysat_t11p_probe_o128 "$seed" \
                "anysat_base-T11P-anysat_t11p_probe_o128-train-${seed}-v3"
            run_named anysat_base anysat_t11p_ft_last3_wide "$seed" \
                "anysat_base-T11P-anysat_t11p_ft_last3_wide-train-${seed}-v3"
            run_named anysat_base anysat_t11p_ft_last3_wide_distill "$seed" \
                "anysat_base-T11P-anysat_t11p_ft_last3_wide_distill-train-${seed}-v3"
        done
        unset ANYSAT_T11P_BATCH ANYSAT_T11P_ACCUM
        ;;
    T11PWIDE)
        # Wide-head B-stage pairs for Clay and Prithvi on the fixed T11P
        # manifest. The 128-d B6F/B6D cells stay untouched; these runs form the
        # B6F1024/B6D1024 tracks so they can be compared with AnySat's
        # B6FAW/B6DAW. Prithvi uses micro-batch 1 x accumulation 2 (effective
        # batch 2); Clay uses two-patch chunks, one optimizer step per chunk.
        # Clay run ids carry -v2: d02ee35 fixed the wide-head trainer that
        # consumed four patches per update (364 instead of 728 steps) in the
        # first seed-17 pair, which is superseded. Prithvi's chunking was fine,
        # but 645a476 fixed its wide trainer training on raw DN instead of the
        # normalised input, so its first three-seed batch is superseded too.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        TASK=T11P
        export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11P/3155a8c5e4dc5c005037b0b41e261530c3eb85d48358c118e67f689ff0d6fb96.jsonl
        export PRITHVI_WORKER_BATCH_SIZE=1
        export PRITHVI_WORKER_GRAD_ACCUM=2
        for seed in 17 42 73; do
            run_named clay_v1_5 clay_s2six_ft_last4_raw1024 "$seed" \
                "clay_v1_5-T11P-clay_s2six_ft_last4_raw1024-train-${seed}-v2"
            run_named clay_v1_5 clay_s2six_ft_last4_raw1024_distill "$seed" \
                "clay_v1_5-T11P-clay_s2six_ft_last4_raw1024_distill-train-${seed}-v2"
            run_named prithvi_eo_v2_300_tl prithvi_t11p_ft_last4_raw1024 "$seed" \
                "prithvi_eo_v2_300_tl-T11P-prithvi_t11p_ft_last4_raw1024-train-${seed}-v2"
            run_named prithvi_eo_v2_300_tl prithvi_t11p_ft_last4_raw1024_distill "$seed" \
                "prithvi_eo_v2_300_tl-T11P-prithvi_t11p_ft_last4_raw1024_distill-train-${seed}-v2"
        done
        unset PRITHVI_WORKER_BATCH_SIZE PRITHVI_WORKER_GRAD_ACCUM
        ;;
    T11DMSE)
        # MSE-vs-cosine distillation control (diagnostic, T11D). Distill before
        # probe for each seed: the probe loads the projector published by the
        # distill run. New target names (oe64mse/aef64mse) keep the cosine
        # projectors untouched.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        TASK=T11D
        export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11D/8e674fc75f1b878cf4688a323666e68601b6ffb1baa942bac6029795b45541a6.jsonl
        for seed in 17 42 73; do
            for teacher in o64 a64; do
                run_named clay_v1_5 "clay_rgbn_distill_${teacher}_mse" "$seed" \
                    "clay_v1_5-T11D-clay_rgbn_distill_${teacher}_mse-train-${seed}"
                run_named clay_v1_5 "clay_rgbn_probe_${teacher}_mse" "$seed" \
                    "clay_v1_5-T11D-clay_rgbn_probe_${teacher}_mse-train-${seed}"
            done
        done
        ;;
    T11PCTRL)
        # Distillation closing controls for T11P (diagnostic):
        # A. OE128 MSE vs cosine distillation (three backbones)
        # B. train-only PCA128 probes (three backbones)
        # C. 10% label wide-head B stage (three backbones; AnySat needs its
        #    10% raw probe to publish the budget-scoped 6->10 adapter).
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        TASK=T11P
        export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11P/3155a8c5e4dc5c005037b0b41e261530c3eb85d48358c118e67f689ff0d6fb96.jsonl
        export PRITHVI_WORKER_BATCH_SIZE=1
        export PRITHVI_WORKER_GRAD_ACCUM=2
        export ANYSAT_T11P_BATCH=1
        export ANYSAT_T11P_ACCUM=2
        for seed in 17 42 73; do
            run_named clay_v1_5 clay_s2six_distill_o128_mse "$seed" \
                "clay_v1_5-T11P-clay_s2six_distill_o128_mse-train-${seed}"
            run_named clay_v1_5 clay_s2six_probe_o128_mse "$seed" \
                "clay_v1_5-T11P-clay_s2six_probe_o128_mse-train-${seed}"
            run_named prithvi_eo_v2_300_tl prithvi_t11p_distill_o128_mse "$seed" \
                "prithvi_eo_v2_300_tl-T11P-prithvi_t11p_distill_o128_mse-train-${seed}"
            run_named prithvi_eo_v2_300_tl prithvi_t11p_probe_o128_mse "$seed" \
                "prithvi_eo_v2_300_tl-T11P-prithvi_t11p_probe_o128_mse-train-${seed}"
            run_named anysat_base anysat_t11p_distill_o128_mse "$seed" \
                "anysat_base-T11P-anysat_t11p_distill_o128_mse-train-${seed}"
            run_named anysat_base anysat_t11p_probe_o128_mse "$seed" \
                "anysat_base-T11P-anysat_t11p_probe_o128_mse-train-${seed}"
            run_named clay_v1_5 clay_s2six_probe_pca128 "$seed" \
                "clay_v1_5-T11P-clay_s2six_probe_pca128-train-${seed}"
            run_named prithvi_eo_v2_300_tl prithvi_t11p_probe_pca128 "$seed" \
                "prithvi_eo_v2_300_tl-T11P-prithvi_t11p_probe_pca128-train-${seed}"
            run_named anysat_base anysat_t11p_probe_pca128 "$seed" \
                "anysat_base-T11P-anysat_t11p_probe_pca128-train-${seed}"
        done
        # C. 10% label panel, fixed 146 train patches from the T11D subset.
        export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11P_subsets/10pct-groups-from-T11D.jsonl
        export RS_EVAL_LABEL_FRACTION=0.1
        for seed in 17 42 73; do
            run_named clay_v1_5 clay_s2six_ft_last4_raw1024 "$seed" \
                "clay_v1_5-T11P-clay_s2six_ft_last4_raw1024-train-${seed}-lf10"
            run_named clay_v1_5 clay_s2six_ft_last4_raw1024_distill "$seed" \
                "clay_v1_5-T11P-clay_s2six_ft_last4_raw1024_distill-train-${seed}-lf10"
            run_named prithvi_eo_v2_300_tl prithvi_t11p_ft_last4_raw1024 "$seed" \
                "prithvi_eo_v2_300_tl-T11P-prithvi_t11p_ft_last4_raw1024-train-${seed}-lf10"
            run_named prithvi_eo_v2_300_tl prithvi_t11p_ft_last4_raw1024_distill "$seed" \
                "prithvi_eo_v2_300_tl-T11P-prithvi_t11p_ft_last4_raw1024_distill-train-${seed}-lf10"
            run_named anysat_base anysat_t11p_probe_raw "$seed" \
                "anysat_base-T11P-anysat_t11p_probe_raw-train-${seed}-lf10"
            run_named anysat_base anysat_t11p_ft_last3_wide "$seed" \
                "anysat_base-T11P-anysat_t11p_ft_last3_wide-train-${seed}-lf10"
            run_named anysat_base anysat_t11p_ft_last3_wide_distill "$seed" \
                "anysat_base-T11P-anysat_t11p_ft_last3_wide_distill-train-${seed}-lf10"
        done
        unset RS_EVAL_LABEL_FRACTION PRITHVI_WORKER_BATCH_SIZE PRITHVI_WORKER_GRAD_ACCUM
        unset ANYSAT_T11P_BATCH ANYSAT_T11P_ACCUM
        ;;
    T11DNCE)
        # Symmetric InfoNCE distillation control (diagnostic, T11D). Distill
        # before probe for each seed; oe64nce/aef64nce keep the cosine and MSE
        # projectors untouched.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        TASK=T11D
        export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11D/8e674fc75f1b878cf4688a323666e68601b6ffb1baa942bac6029795b45541a6.jsonl
        for seed in 17 42 73; do
            for teacher in o64 a64; do
                run_named clay_v1_5 "clay_rgbn_distill_${teacher}_nce" "$seed" \
                    "clay_v1_5-T11D-clay_rgbn_distill_${teacher}_nce-train-${seed}"
                run_named clay_v1_5 "clay_rgbn_probe_${teacher}_nce" "$seed" \
                    "clay_v1_5-T11D-clay_rgbn_probe_${teacher}_nce-train-${seed}"
            done
        done
        ;;
    T11PNCE)
        # Six-band replication of the positive T11D InfoNCE result (diagnostic).
        # Distill before probe per backbone and seed; oe128nce keeps the cosine
        # and MSE projectors untouched.
        export RS_EVAL_RERUN_STOP_ON_FAIL=1
        TASK=T11P
        export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11P/3155a8c5e4dc5c005037b0b41e261530c3eb85d48358c118e67f689ff0d6fb96.jsonl
        for seed in 17 42 73; do
            run_named clay_v1_5 clay_s2six_distill_o128_nce "$seed" \
                "clay_v1_5-T11P-clay_s2six_distill_o128_nce-train-${seed}"
            run_named clay_v1_5 clay_s2six_probe_o128_nce "$seed" \
                "clay_v1_5-T11P-clay_s2six_probe_o128_nce-train-${seed}"
            run_named prithvi_eo_v2_300_tl prithvi_t11p_distill_o128_nce "$seed" \
                "prithvi_eo_v2_300_tl-T11P-prithvi_t11p_distill_o128_nce-train-${seed}"
            run_named prithvi_eo_v2_300_tl prithvi_t11p_probe_o128_nce "$seed" \
                "prithvi_eo_v2_300_tl-T11P-prithvi_t11p_probe_o128_nce-train-${seed}"
            run_named anysat_base anysat_t11p_distill_o128_nce "$seed" \
                "anysat_base-T11P-anysat_t11p_distill_o128_nce-train-${seed}"
            run_named anysat_base anysat_t11p_probe_o128_nce "$seed" \
                "anysat_base-T11P-anysat_t11p_probe_o128_nce-train-${seed}"
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
