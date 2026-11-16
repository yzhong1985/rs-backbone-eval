#!/usr/bin/env bash
# One-shot readiness check for a fresh deployment, bare metal or container.
#
#   bash scripts/preflight.sh            # host checks
#   bash scripts/preflight.sh --docker   # also check the image and the toolkit
#
# Never starts a run and never writes to the ledger; it only reports. Exit 1 if
# any check fails, so it can gate a deploy script.
set -uo pipefail
cd "$(dirname "$0")/.."

# The worker checkpoint variables live in .env; sourcing it here is the same
# thing the runtime does, and it keeps --docker/--bare checks identical.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

WANT_DOCKER=0
[ "${1:-}" = "--docker" ] && WANT_DOCKER=1

fail=0
ok() { printf '  ok    %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }
bad() { printf '  FAIL  %s\n' "$*"; fail=$((fail + 1)); }

echo "== toolchain =="
if command -v uv > /dev/null 2>&1; then ok "uv $(uv --version | awk '{print $2}')"
else bad "uv not found (https://astral.sh/uv)"; fi
if uv run python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2> /dev/null; then
    ok "python $(uv run python -c 'import platform; print(platform.python_version())')"
else bad "python 3.11+ not resolvable through uv"; fi
if [ -f .env ]; then ok ".env present"
else warn ".env missing; relying on exported RS_EVAL_* variables"; fi

echo "== runtime paths =="
for pair in \
    "data:RS_EVAL_DATA_ROOT" \
    "weights:RS_EVAL_WEIGHTS_ROOT" \
    "runs:RS_EVAL_RUNS_ROOT" \
    "cache:RS_EVAL_CACHE_ROOT"; do
    label=${pair%%:*}
    var=${pair#*:}
    value=$(uv run python - "$var" <<'PY' 2> /dev/null
import os, sys
from rs_backbone_eval.paths import RuntimePaths
print(getattr(RuntimePaths.from_environment(), sys.argv[1].removeprefix("RS_EVAL_").lower()))
PY
)
    if [ -z "$value" ]; then bad "$var unresolved"; continue; fi
    if [ -d "$value" ]; then ok "$label: $value"
    else bad "$label missing: $value"; fi
done

runs_root=$(uv run python - <<'PY' 2> /dev/null
from rs_backbone_eval.paths import RuntimePaths
print(RuntimePaths.from_environment().runs_root)
PY
)
if [ -n "$runs_root" ] && [ -w "$(dirname "$runs_root")" ]; then ok "runs parent writable"
elif [ -n "$runs_root" ]; then bad "cannot write beside $runs_root"; fi

echo "== data =="
if ls "$(uv run python -c 'from rs_backbone_eval.paths import RuntimePaths as R; print(R.from_environment().data_root)' 2> /dev/null)"/manifests/*/*.jsonl > /dev/null 2>&1; then
    ok "at least one manifest"
else
    warn "no manifests yet; run: uv run rs-eval build-manifest --task T11D"
fi
for pair in \
    "CLAY_CHECKPOINT:clay v1.5" \
    "PRITHVI_CHECKPOINT:prithvi eo-v2-300m" \
    "ANYSAT_CHECKPOINT:anysat base" \
    "LIM_CHECKPOINT:lim mit_huge"; do
    var=${pair%%:*}
    value=${!var:-}
    if [ -z "$value" ]; then warn "$var unset"; elif [ -e "$value" ]; then ok "${pair#*:} weight"; else bad "${pair#*:} weight missing: $value"; fi
done

echo "== gpu =="
if command -v nvidia-smi > /dev/null 2>&1; then
    read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | tr ',' ' ')
    ok "gpu present (${used}/${total} MiB used)"
else
    warn "no nvidia-smi; CPU-only cells still work"
fi

echo "== queue =="
queue_root=$(uv run python - <<'PY' 2> /dev/null
from rs_backbone_eval.paths import RuntimePaths
from rs_backbone_eval.queue import default_queue_root
print(default_queue_root(RuntimePaths.from_environment()))
PY
)
if [ -n "$queue_root" ]; then
    if mkdir -p "$queue_root" 2> /dev/null; then ok "queue: $queue_root"
    else bad "queue root not creatable: $queue_root"; fi
fi

if [ "$WANT_DOCKER" = 1 ]; then
    echo "== docker =="
    if docker info > /dev/null 2>&1; then ok "docker daemon reachable"
    else bad "docker daemon not reachable"; fi
    if docker info --format '{{json .Runtimes}}' 2> /dev/null | grep -q nvidia; then
        ok "nvidia runtime registered"
    else
        warn "nvidia runtime not registered; install nvidia-container-toolkit"
    fi
    if docker image inspect rs-eval:local > /dev/null 2>&1; then ok "image rs-eval:local built"
    else warn "image rs-eval:local missing; docker compose -f docker/compose.yaml build"; fi
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "preflight: passed"
    exit 0
fi
echo "preflight: ${fail} failure(s)"
exit 1
