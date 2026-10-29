#!/usr/bin/env bash
# Fetch the FAIR1M scenes used by T10 from a public mirror, split them into
# train / val / unlabeled, and verify that every image pairs with an annotation.
#
# Why a mirror: gaofen-challenge.com serves a JavaScript shell with no direct
# link, and huggingface.co is unreachable from the evaluation host. The two
# repositories below are reachable and were verified on 2026-10-29.
#
#   LittleCollections/FAIR1M1  images + labelXmls  -> 14,756 train pairs, plus
#                                                   1,732 images with no XML
#   LittleCollections/FAIR1M2  images + labelXmls  ->  8,287 pairs, of which
#                                                   1,732 are the val scenes
#
# The 1,732 unpaired images in FAIR1M1 are NOT the val set, even though the ids
# look like they should be: pairing them with FAIR1M2's val labels (the obvious
# shortcut) leaves 74% of the pairs with mismatched sizes and unusable boxes
# (review 2026-10-29). This script therefore takes each image from the repo that
# ships its own label and refuses to finish unless the sizes agree.
#
# Usage:
#   bash scripts/fetch_fair1m.sh [DATA_ROOT]
#
# DATA_ROOT defaults to RS_EVAL_DATA_ROOT, then to RuntimePaths (.env).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_root="${1:-${RS_EVAL_DATA_ROOT:-}}"
if [ -z "$data_root" ]; then
  data_root="$(uv run python -c 'from rs_backbone_eval.paths import RuntimePaths; print(RuntimePaths.from_environment().data_root)')"
fi
raw="$data_root/raw/fair1m"
downloads="$raw/downloads"
mirror="${FAIR1M_MIRROR:-https://hf-mirror.com/datasets}"
echo "data root: $data_root"
echo "mirror:    $mirror"

mkdir -p "$downloads" "$raw/provenance"

fetch() {
  local dataset="$1" name="$2"
  local dest="$downloads/$name"
  if [ -f "$dest" ] && unzip -tqq "$dest" >/dev/null 2>&1; then
    echo "ok       $name (already present)"
  else
    echo "fetching $dataset/$name"
    curl -sL --fail --retry 5 --retry-delay 5 -C - -o "$dest" "$mirror/$dataset/$name"
  fi
  unzip -tqq "$dest" >/dev/null || { echo "$name is not a valid zip" >&2; return 1; }
}

fetch LittleCollections/FAIR1M1 images.zip
fetch LittleCollections/FAIR1M1 images-1.zip
fetch LittleCollections/FAIR1M1 images-2.zip
fetch LittleCollections/FAIR1M1 labelXmls.zip
fetch LittleCollections/FAIR1M2 images.zip
fetch LittleCollections/FAIR1M2 labelXmls.zip

sha256sum "$downloads"/*.zip | tee "$raw/provenance/sha256sums.txt"

scratch="$(mktemp -d "$raw/.extract.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

for name in images.zip images-1.zip images-2.zip; do
  echo "extracting $name"
  unzip -q -o "$downloads/$name" -d "$scratch/train"
done
echo "extracting FAIR1M1 labelXmls.zip"
unzip -q -o "$downloads/labelXmls.zip" -d "$scratch/train"

echo "extracting FAIR1M2 images.zip"
unzip -q -o "$downloads/fair1m2_images.zip" -d "$scratch/val"
echo "extracting FAIR1M2 labelXmls.zip"
unzip -q -o "$downloads/fair1m2_labelXmls.zip" -d "$scratch/val"

# Pairing, splitting and the size guard all live in one Python block: the rule
# ("each image comes from the repo that also ships its label") is easier to
# enforce here than in shell.
uv run python - "$scratch" "$raw" <<'PY'
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from rs_backbone_eval.data.image_header import image_size

scratch, raw = Path(sys.argv[1]), Path(sys.argv[2])
xml_size = lambda xml: (  # noqa: E731 - two-line helper, used twice
    int(float(ET.parse(xml).getroot().findtext("size/width"))),
    int(float(ET.parse(xml).getroot().findtext("size/height"))),
)


def index(root: Path, suffix: str) -> dict[str, Path]:
    return {path.stem: path for path in root.rglob(f"*{suffix}")}


train_images = index(scratch / "train", ".tif")
train_xmls = index(scratch / "train", ".xml")
val_images = index(scratch / "val", ".tif")
val_xmls = index(scratch / "val", ".xml")

train_stems = sorted(set(train_images) & set(train_xmls))
unlabeled = sorted(set(train_images) - set(train_xmls))
# The val scenes are the FAIR1M2 pairs that are not part of train.
val_stems = sorted(set(val_images) & set(val_xmls) - set(train_xmls))

for name in ("images/train", "images/val", "images/unlabeled", "labelXml", "labelXml/val"):
    (raw / name).mkdir(parents=True, exist_ok=True)

for stem in train_stems:
    shutil.copy2(train_images[stem], raw / "images/train" / f"{stem}.tif")
    shutil.copy2(train_xmls[stem], raw / "labelXml" / f"{stem}.xml")
for stem in val_stems:
    shutil.copy2(val_images[stem], raw / "images/val" / f"{stem}.tif")
    shutil.copy2(val_xmls[stem], raw / "labelXml/val" / f"{stem}.xml")
for stem in unlabeled:
    shutil.copy2(train_images[stem], raw / "images/unlabeled" / f"{stem}.tif")

print(f"train pairs {len(train_stems)}, val pairs {len(val_stems)}, "
      f"unlabeled images {len(unlabeled)}")

# The guard that the manual pairing lacked: an image and its annotation must
# describe the same rectangle, or every box in that scene is wrong.
for split, images, xmls, stems in (
    ("train", raw / "images/train", raw / "labelXml", train_stems),
    ("val", raw / "images/val", raw / "labelXml/val", val_stems),
):
    match = 0
    for stem in stems:
        xml = xmls / f"{stem}.xml"
        if image_size(images / f"{stem}.tif") == xml_size(xml):
            match += 1
    share = match / max(len(stems), 1)
    print(f"{split}: sizes agree for {match}/{len(stems)} ({share:.2%})")
    if share < 0.99:
        raise SystemExit(
            f"{split}: image and annotation sizes disagree for {len(stems) - match} scenes; "
            "this pairing is wrong, do not build a manifest from it"
        )
PY

echo
echo "Source note: FAIR1M-1.0 train is identical to 2.0 train, so it is used for"
echo "T10; official 2.0 val/test are not part of this download. Runs built from it"
echo "are FAIR1M-train-holdout and must say so."
