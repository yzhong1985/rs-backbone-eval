#!/usr/bin/env bash
# FBP 2017 embedding extraction: labels first, then the AEF and OlmoEarth lanes.
#
# The two archives are on different hosts and have very different throughput
# (measured 2026-11-11): OlmoEarth ~2.3 MB/s and 33 s per 512 window, AEF
# ~0.3 MB/s and 10 s per 256 tile (a 512 read is 12m44 because the COG blocks
# are 1024 px). Running both lanes at once is therefore fine, but the AEF lane
# must stay at one worker: four parallel reads measured *slower* per byte.
#
# Phases are resumable: an existing per-scene .npz is skipped, so re-running
# after an interruption continues where it stopped.
#
# Usage:  bash scripts/fbp_extract_all.sh [out_dir]
# Logs:   ${LOGDIR:-/tmp/opencode}/fbp_{labels,aef,olmo,summary}.log
set -euo pipefail

cd "$(dirname "$0")/.."

OUT=${1:-/home/yzhong/rs-eval/rs-eval-cache/fbp2017}
LOGDIR=${LOGDIR:-/tmp/opencode}
PY=workers/prithvi/.venv/bin/python
LABEL_WORKERS=${LABEL_WORKERS:-4}
AEF_WORKERS=${AEF_WORKERS:-1}
OLMO_WORKERS=${OLMO_WORKERS:-4}

mkdir -p "$OUT" "$LOGDIR"

echo "[$(date -Is)] labels -> $OUT (workers=$LABEL_WORKERS)"
"$PY" scripts/fbp_extract_windows.py --phase labels --out "$OUT" \
    --workers "$LABEL_WORKERS" > "$LOGDIR/fbp_labels.log" 2>&1

echo "[$(date -Is)] aef (workers=$AEF_WORKERS) and olmo (workers=$OLMO_WORKERS) in parallel"
"$PY" scripts/fbp_extract_windows.py --phase aef --out "$OUT" \
    --workers "$AEF_WORKERS" > "$LOGDIR/fbp_aef.log" 2>&1 &
aef_pid=$!

uv run --python 3.12 --with 'zarr>=3' --with obstore --with numpy \
    python scripts/fbp_extract_windows.py --phase olmo --out "$OUT" \
    --workers "$OLMO_WORKERS" > "$LOGDIR/fbp_olmo.log" 2>&1 &
olmo_pid=$!

status=0
wait "$aef_pid" || status=1
wait "$olmo_pid" || status=1

echo "[$(date -Is)] summary"
"$PY" scripts/fbp_extract_windows.py --phase summary --out "$OUT" \
    > "$LOGDIR/fbp_summary.log" 2>&1 || status=1
cat "$LOGDIR/fbp_summary.log"

echo "[$(date -Is)] done (status=$status)"
exit "$status"
