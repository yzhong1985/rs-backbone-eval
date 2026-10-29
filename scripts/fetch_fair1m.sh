#!/usr/bin/env bash
# Fetch the FAIR1M scenes used by T10 (detection) from a public mirror
# and verify it pairs image-for-image with its annotations.
#
# Why not the official site: gaofen-challenge.com serves a JavaScript shell with
# no direct link, and huggingface.co is unreachable from this host. The mirror
# below is reachable and was verified on 2026-10-29; it carries the **1.0 train**
# set, which the official FAIR1M-2.0 release keeps unchanged. It is therefore a
# legitimate source for the internal `train-holdout` protocol, and it is NOT the
# official 2.0 val/test — those are never claimed by a run from this data.
#
# Usage:
#   bash scripts/fetch_fair1m.sh [DATA_ROOT]
#
# DATA_ROOT defaults to RS_EVAL_DATA_ROOT, then to RuntimePaths (.env). Files land
# under $DATA_ROOT/raw/fair1m/{images/train,labelXml}; re-running skips archives
# whose checksum already matches and records the checksums it verified.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_root="${1:-${RS_EVAL_DATA_ROOT:-}}"
if [ -z "$data_root" ]; then
  data_root="$(uv run python -c 'from rs_backbone_eval.paths import RuntimePaths; print(RuntimePaths.from_environment().data_root)')"
fi
raw="$data_root/raw/fair1m"
downloads="$data_root/raw/fair1m/downloads"
mirror="${FAIR1M_MIRROR:-https://hf-mirror.com/datasets/LittleCollections/FAIR1M1/resolve/main}"
echo "data root: $data_root"
echo "mirror:    $mirror"

mkdir -p "$downloads" "$raw/images/train" "$raw/labelXml" "$raw/provenance"

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# fetch <filename>
fetch() {
  local name="$1" dest="$downloads/$1"
  if [ -f "$dest" ] && unzip -tqq "$dest" >/dev/null 2>&1; then
    echo "ok       $name (already present and valid)"
  else
    echo "fetching $name"
    curl -sL --fail --retry 5 --retry-delay 5 -C - -o "$dest" "$mirror/$name"
  fi
  unzip -tqq "$dest" >/dev/null || { echo "$name is not a valid zip" >&2; return 1; }
}

for name in images.zip images-1.zip images-2.zip labelXmls.zip; do
  fetch "$name"
done

sha256sum "$downloads"/*.zip | tee "$raw/provenance/sha256sums.txt"

# Extract into a scratch directory, then move the payload into place, so a
# half-written archive cannot leave a mixture of two splits.
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

for name in images.zip images-1.zip images-2.zip; do
  echo "extracting $name"
  unzip -q -o "$downloads/$name" -d "$scratch"
done
find "$scratch" -type f \( -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) \
  -exec mv -n {} "$raw/images/train/" +

echo "extracting labelXmls.zip"
unzip -q -o "$downloads/labelXmls.zip" -d "$scratch"
find "$scratch" -type f -iname '*.xml' -exec mv -n {} "$raw/labelXml/" +

images=$(find "$raw/images/train" -maxdepth 1 -type f | wc -l)
labels=$(find "$raw/labelXml" -maxdepth 1 -type f -name '*.xml' | wc -l)
echo "images: $images"
echo "labels: $labels"
if [ "$images" -ne "$labels" ]; then
  echo "image/annotation count mismatch; every image needs exactly one XML" >&2
  exit 1
fi

echo
echo "Source note: this is the FAIR1M 1.0 train set (identical to the 2.0 train"
echo "set). It is not the official 2.0 val/test; runs built from it are"
echo "FAIR1M-train-holdout and must say so."
