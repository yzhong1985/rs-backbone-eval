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
set -uos pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
. ./.env
set +a

WORKERS="${1:-T3 T5 T1}"
MAXPAR="${RS_COLAB_PARALLEL:-8}"
OSROOT="$RS_EVAL_DATA_ROOT/raw"
echo "bootstrap for: $WORKERS (data root $RS_EVAL_DATA_ROOT)"

fetch() {  # url dest
    echo "GET $1"
    wget -q -c "$1" -O "$2" || { echo "FAILED $1"; return 1; }
} # no checksum here: the catalog's sha256 audit runs on the main machine

count_tiffs() { find "$1" -name '*.tif*' | wc -l; }

for task in $WORKERS; do
    case "$task" in
        T5)
            root="$OSROOT/multi-temporal-crop-classification"
            if [ -d "$root/training_chips" ] && [ -d "$root/validation_chips" ]; then
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
            if [ -d "$root/extracted/pre" ]; then
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
            if [ -d "$root/v1.1/data" ]; then
                echo "T1 layout present; skip"
                continue
            fi
            mkdir -p "$root/v1.1"
            # every relative path below raw/sen1floods11 in the manifest is an
            # object of the same name in the public GCS bucket
            python3 - "$manifest" "$root/v1.1" <<'PYEOF'
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
with open("/tmp/sen1floods11_urls.txt", "w", encoding="utf-8") as out:
    for relative in sorted(wanted):
        prefix = "raw/sen1floods11/"
        assert relative.startswith(prefix), relative
        tail = relative[len(prefix):]
        out.write(f"{pub}{tail}\n{target}/{tail}\n")
print(f"{len(wanted)} T1 files listed")
PYEOF
            while read -r url dest; do
                [ -s "$dest" ] && continue
                mkdir -p "$(dirname "$dest")"
                echo "$url $dest"
            done < /tmp/sen1floods11_urls.txt > /tmp/sen1floods11_todo.txt
            wc -l < /tmp/sen1floods11_todo.txt | xargs echo "T1 to fetch:"
            xargs -a /tmp/sen1floods11_todo.txt -n2 -P"$MAXPAR" sh -c 'wget -q -c "$1" -O "$2" || echo "FAILED $1"' _
            rm -f /tmp/sen1floods11_urls.txt /tmp/sen1floods11_todo.txt
            echo "T1 fetched: $(count_tiffs "$root") tiffs"
            ;;
        *) echo "unknown task $task"; exit 2 ;;
    esac
done
echo "BOOTSTRAP-OK"
