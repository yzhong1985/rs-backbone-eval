#!/bin/bash
# Fetch WHU-OPT-SAR (GF-1 optical + GF-3 SAR, 100 sheets) from the hf-mirror
# copy of the official release, verify sizes/checksums and unpack the unique
# members.
#
# Why the mirror: the official Google Drive folder is only reachable through a
# SOCKS proxy on this host and moves ~0 B/s, while hf-mirror.com runs at
# 70-75 MB/s. The mirror dataset states the upstream link
# (https://github.com/AmberHen/WHU-OPT-SAR-dataset) in its card.
#
# The archives each contain 200 tifs: 100 sheets plus a "(1)" duplicate of every
# sheet (Drive sync artefact). Only the non-duplicate members are extracted.
#
# Layout: <root>/{optical,sar,lbl}/NH49E004021.tif ... (100 each, 5556x3704).
set -euo pipefail

ROOT="${WHU_ROOT:-/home/yzhong/rs-eval-data/whu-opt-sar}"
BASE="${WHU_MIRROR:-https://hf-mirror.com/datasets/125oii/whu_opt_sar/resolve/main}"
mkdir -p "$ROOT"
cd "$ROOT"

declare -A SHA=(
  [optical.zip]="a1a5dcfff2ec7274ba0576c99bfe60f491aa2bf2ecd34325f212ec21f8a35bd9"
  [sar.zip]="5c71d6ca7d50bbbc2f0ca12b475115c07395e17a55388015993dd57d93b7047d"
  [lbl.zip]="f9b3b281f490f9442b28ce28c1d209ad4448b6cb8c9cb83fa9dc8304d03464c1"
)

for f in optical.zip sar.zip lbl.zip; do
  if [ -f "$f" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "${SHA[$f]}" ]; then
    echo "have $f"
    continue
  fi
  echo "=== download $f"
  curl -fSL --continue-at - -o "$f" "$BASE/$f"
  actual="$(sha256sum "$f" | cut -d' ' -f1)"
  if [ "$actual" != "${SHA[$f]}" ]; then
    echo "checksum mismatch for $f: $actual" >&2
    exit 3
  fi
done

python3 - <<'PY'
import zipfile
for archive in ("optical.zip", "sar.zip", "lbl.zip"):
    with zipfile.ZipFile(archive) as z:
        members = [i for i in z.infolist() if not i.is_dir() and "(1)" not in i.filename]
        z.extractall(".", members=members)
        print(f"{archive}: extracted {len(members)} members")
PY

for d in optical sar lbl; do
  count=$(find "$d" -name '*.tif' | wc -l)
  echo "$d: $count tifs"
  [ "$count" -eq 100 ] || { echo "expected 100 tifs in $d" >&2; exit 4; }
done
du -sh optical sar lbl
