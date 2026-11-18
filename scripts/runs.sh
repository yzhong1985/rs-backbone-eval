#!/usr/bin/env bash
# 手动管理 run：入队、查看、看日志、停止。账本仍由 rs-eval / worker 写，本脚本不改 run.json。
#
#   scripts/runs.sh enqueue configs/runs/<spec>.yaml [--dry-run]
#   scripts/runs.sh status
#   scripts/runs.sh logs <run_id>
#   scripts/runs.sh stop <run_id>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_ROOT="${RS_EVAL_RUNS_ROOT:-/geodata/rs-eval/runs}"
QUEUE_ROOT="${RS_EVAL_QUEUE_ROOT:-/geodata/rs-eval/queue}"
cd "$ROOT"

die() { echo "error: $*" >&2; exit 1; }

# 读取 spec，输出 rs-eval queue add 的参数，每行一个，便于 mapfile
spec_args() {
  uv run python -I - "$1" <<'PY'
import shlex, sys, yaml
spec = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
required = ("task", "model", "protocol", "seed")
missing = [k for k in required if k not in spec]
if missing:
    sys.exit(f"spec missing: {', '.join(missing)}")
args = ["--task", str(spec["task"]), "--model", str(spec["model"]),
        "--protocol", str(spec["protocol"]), "--seed", str(spec["seed"])]
if "action" in spec:
    args += ["--action", str(spec["action"])]
for key, value in (spec.get("env") or {}).items():
    args += ["--env", f"{key}={value}"]
if "note" in spec:
    args += ["--note", str(spec["note"])]
print("\n".join(args))
PY
}

cmd_enqueue() {
  local spec="${1:-}"; shift || true
  [[ -f "$spec" ]] || die "no such spec: $spec"
  local dry=0
  [[ "${1:-}" == "--dry-run" ]] && dry=1
  mapfile -t args < <(spec_args "$spec")
  if [[ $dry -eq 1 ]]; then
    printf 'rs-eval queue add %s\n' "$(printf '%q ' "${args[@]}")"
    exit 0
  fi
  uv run rs-eval queue add "${args[@]}"
}

cmd_status() {
  echo "== queue"
  uv run rs-eval queue list 2>&1 | tail -n +1
  echo "== workers"
  pgrep -af "lim_worker/runner.py|workers/.*/runner.py" || echo "(none)"
  echo "== gpu"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
}

cmd_logs() {
  local run_id="${1:-}"
  [[ -n "$run_id" ]] || die "usage: logs <run_id>"
  local log="$RUNS_ROOT/$run_id/train.log"
  [[ -f "$log" ]] || die "no train.log for $run_id (AGENTS.md: collect_run_logs.sh mirrors it)"
  tail -n 40 "$log"
}

cmd_stop() {
  local run_id="${1:-}"
  [[ -n "$run_id" ]] || die "usage: stop <run_id>"
  local pids
  pids="$(pgrep -f "runner.py .*${run_id}/request.json" || true)"
  if [[ -n "$pids" ]]; then
    echo "stopping worker(s): $pids"
    kill $pids
    echo "worker stopped; the queue record stays in running/ until the consumer reports it"
    echo "check: scripts/runs.sh status"
    exit 0
  fi
  if ls "$QUEUE_ROOT/pending/" | grep -q -- "-${run_id}\.json$"; then
    uv run rs-eval queue cancel "$run_id" --reason "stopped by hand"
  else
    die "$run_id is not a running worker or a pending job"
  fi
}

case "${1:-}" in
  enqueue) shift; cmd_enqueue "$@" ;;
  status)  shift; cmd_status ;;
  logs)    shift; cmd_logs "$@" ;;
  stop)    shift; cmd_stop "$@" ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
