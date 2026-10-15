#!/usr/bin/env bash
# Colab data bootstrap: builds the RS_EVAL_DATA_ROOT layout from public sources.
#
# Public preregs: the NVIDIA/IBM HF dataset archives (T5, T3) and the public
# sen1floods11 GCS bucket (T1). The manifest rows are NOT rebuilt here; the
# packed manifests (data_root/manifests) reference the same relative paths the
# archives extract to, so only the layout matters. Idempotent: everything
# already present is skipped. Run on a machine with fast egress (Colab).
#
# Usage: bash scripts/colab_bootstrap.sh [task ids]   # default: T3 T5 T1
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
. ./.env
set +a

WORKERS="${*:-T3 T5 T1}"
MAXPAR="${RS_COLAB_PARALLEL:-8}"
OSROOT="$RS_EVAL_DATA_ROOT/raw"
echo "bootstrap for: $WORKERS (data root $RS_EVAL_DATA_ROOT)"

fetch() {  # url dest
    echo "GET $1"
    wget -q -c "$1" -O "$2.part" || { echo "FAILED $1"; return 1; }
    mv "$2.part" "$2"
} # no checksum here: the catalog's sha256 audit runs on the main machine

count_tiffs() { find "$1" -name '*.tif*' | wc -l; }

verify_task() {
    python3 - "$RS_EVAL_DATA_ROOT" "$1" <<'PYEOF'
import json, pathlib, sys
root, task = pathlib.Path(sys.argv[1]), sys.argv[2]
manifests = sorted((root / "manifests" / task).glob("*.jsonl"))
if not manifests:
    sys.exit(f"no {task} manifests; unpack the shared manifests first")
count = 0
for manifest in manifests:
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for relative in [*row["image_paths"], row["label_path"]]:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                sys.exit(f"missing or empty manifest file: {path}")
        count += 1
if not count:
    sys.exit(f"empty {task} manifests")
print(f"{task}: verified paths for {count} manifest rows")
PYEOF
}

for task in $WORKERS; do
    case "$task" in
        T5)
            root="$OSROOT/multi-temporal-crop-classification"
            if [ -d "$root/training_chips" ] && [ -d "$root/validation_chips" ] && verify_task T5; then
                echo "T5 layout present ($(count_tiffs "$root") tiffs); skip"
                continue
            fi
            mkdir -p "$root/main"
            fetch https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification/resolve/main/training_chips.tgz "$root/main/training_chips.tgz"
            fetch https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification/resolve/main/validation_chips.tgz "$root/main/validation_chips.tgz"
            tar xzf "$root/main/training_chips.tgz" --exclude='._*' -C "$root"
            tar xzf "$root/main/validation_chips.tgz" --exclude='._*' -C "$root"
            rm -f "$root/main/training_chips.tgz" "$root/main/validation_chips.tgz"
            echo "T5 extracted: $(count_tiffs "$root/training_chips") train + $(count_tiffs "$root/validation_chips") validation tiffs"
            ;;
        T3)
            root="$OSROOT/burn_intensity"
            if [ -d "$root/extracted/pre" ] && verify_task T3; then
                echo "T3 layout present ($(count_tiffs "$root/extracted") tiffs); skip"
                continue
            fi
            mkdir -p "$root/main" "$root/extracted"
            for step in pre during post; do
                fetch "https://huggingface.co/datasets/ibm-nasa-geospatial/burn_intensity/resolve/main/$step.zip" "$root/main/$step.zip"
                unzip -oq "$root/main/$step.zip" -d "$root/extracted"
                rm -f "$root/main/$step.zip"
            done
            echo "T3 extracted: $(count_tiffs "$root/extracted") tiffs"
            ;;
        T1)
            root="$OSROOT/sen1floods11"
            manifest="$RS_EVAL_DATA_ROOT/manifests/T1"
            if [ -d "$root/v1.1/data" ] && verify_task T1; then
                echo "T1 layout present; skip"
                continue
            fi
            mkdir -p "$root/v1.1"
            # every relative path below raw/sen1floods11 in the manifest is an
            # object of the same name in the public GCS bucket
            queue="$(mktemp)"
            trap 'rm -f "$queue"' EXIT
            python3 - "$manifest" "$root" "$queue" <<'PYEOF'
import json, sys, pathlib
manifest_dir, target = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = sorted(pathlib.Path(manifest_dir).glob("*.jsonl")) or []
if not rows:
    sys.exit("no T1 manifest found; mount data_root with manifests/ first")
wanted = set()
for path in rows:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        wanted.update(row["image_paths"])
        wanted.add(row["label_path"])
pub = "https://storage.googleapis.com/sen1floods11/"
with open(sys.argv[3], "wb") as out:
    for relative in sorted(wanted):
        prefix = "raw/sen1floods11/"
        assert relative.startswith(prefix), relative
        tail = relative[len(prefix):]
        destination = target / tail
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        out.write(f"{pub}{tail}\0{destination}\0".encode())
print(f"{len(wanted)} T1 files listed")
PYEOF
            xargs -0 -r -a "$queue" -n2 -P"$MAXPAR" sh -c \
                'wget -q -c "$1" -O "$2.part" && mv "$2.part" "$2"' _
            rm -f "$queue"
            trap - EXIT
            echo "T1 fetched: $(count_tiffs "$root") tiffs"
            ;;
        *) echo "unknown task $task"; exit 2 ;;
    esac
    verify_task "$task"
done
echo "BOOTSTRAP-OK"
