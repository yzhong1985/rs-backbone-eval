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
        echo "$(date -Is) OK $rid $(grep -oE 'water_iou=[0-9.]+|mean_iou=[0-9.]+' "$rlog" | tail -1)"
    else
        failed=$((failed + 1))
        echo "$(date -Is) FAIL $rid rc=$? (see $rlog)"
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
        echo "$(date -Is) OK $rid $(grep -oE 'water_iou=[0-9.]+|mean_iou=[0-9.]+' "$rlog" | tail -1)"
    else
        failed=$((failed + 1))
        echo "$(date -Is) FAIL $rid rc=$? (see $rlog)"
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
    *)
        echo "unknown task: $TASK"; exit 2 ;;
esac

echo "$(date -Is) === $TASK done: $succeeded ok, $failed failed ==="
if [ "$failed" -gt 0 ]; then
    echo "$(date -Is) FAILURES: $failed run(s) failed" >&2
    exit 1
fi
exit 0
