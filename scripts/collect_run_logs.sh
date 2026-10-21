#!/usr/bin/env bash
# Mirror the dispatcher's per-run training logs into the run directories.
#
# The dispatcher writes each run's stdout/stderr to $LOGDIR/<run_id>.log, which
# lives on /tmp and is lost on reboot. The ledger is the durable record, so copy
# the log next to run.json. Idempotent: a log already mirrored unchanged is left
# alone, and a log that is still growing is copied on the next pass.
#
# Usage: scripts/collect_run_logs.sh [logdir] [runs_root]
set -uo pipefail
cd "$(dirname "$0")/.."

set -a
. ./.env
set +a
LOGDIR="${1:-${RS_EVAL_RERUN_LOGDIR:-/tmp/opencode}}"
RUNS_ROOT="${2:-${RS_EVAL_RUNS_ROOT:-/geodata/rs-eval/runs}}"

copied=0
skipped=0
missing=0
for run_json in "$RUNS_ROOT"/*/run.json; do
    [ -f "$run_json" ] || continue
    run_dir=$(dirname "$run_json")
    run_id=$(basename "$run_dir")
    log="$LOGDIR/$run_id.log"
    target="$run_dir/train.log"
    if [ ! -f "$log" ]; then
        missing=$((missing + 1))
        continue
    fi
    if [ -f "$target" ] && cmp -s "$log" "$target"; then
        skipped=$((skipped + 1))
        continue
    fi
    cp "$log" "$target" && copied=$((copied + 1))
done

echo "$(date -Is) logs: copied $copied, unchanged $skipped, no log $missing (logdir $LOGDIR)"
