#!/usr/bin/env bash
# Fetch iSAID (annotations + semantic masks) and the DOTA-v1.0 images iSAID
# reuses. iSAID ships labels only, so the imagery comes from DOTA train/val.
#
# Why a script instead of `rs-eval download`: every iSAID/DOTA file is hosted on
# Google Drive. Drive answers with an HTML confirmation page for files over
# ~100 MB, so a plain HTTP download (aria2c/python-http) saves a web page, not
# the archive. gdown completes the confirm-token handshake; the catalog entry is
# therefore deferred (see docs/design/iSAID与LIM评测方案 §4.5) and this script is
# the reproducible path.
#
# Usage:
#   bash scripts/fetch_isaid.sh [DATA_ROOT]
#
# DATA_ROOT defaults to RS_EVAL_DATA_ROOT, then to RuntimePaths (i.e. .env).
# Set HTTPS_PROXY/HTTP_PROXY if Drive needs a proxy; nothing is hard-coded.
# Re-running verifies checksums and skips files that already match.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_root="${1:-${RS_EVAL_DATA_ROOT:-}}"
if [ -z "$data_root" ]; then
  data_root="$(uv run python -c 'from rs_backbone_eval.paths import RuntimePaths; print(RuntimePaths.from_environment().data_root)')"
fi
raw="$data_root/raw/isaid"
echo "data root: $data_root"

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# download <drive-id> <relative-path> <sha256>
download() {
  local id="$1" rel="$2" want="$3" dest="$raw/$2"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$want" ]; then
    echo "ok       $rel"
    return 0
  fi
  echo "fetching $rel"
  uvx --from gdown gdown "$id" -O "$dest" --continue
  local got
  got="$(sha256_of "$dest")"
  if [ "$got" != "$want" ]; then
    echo "checksum mismatch for $rel" >&2
    echo "  expected $want" >&2
    echo "  got      $got" >&2
    return 1
  fi
  echo "ok       $rel"
}

# Checksums are the ones recorded in data_root/derived/isaid/p0a_report.json.
download 1-PYSXak2JBg3xuZWzAWVXfKQO1TkPCqF train/iSAID_train.json \
  fce958899207d52e0163f15b0251be191c9d1e3c3ec5cee4060b908cf86ed23e
download 1YLjZ1cmA9PH3OfzMF-eq6T-O9FTGvSrx train/semantic_masks.zip \
  4813569d32be8a956f16223bca4b145c319565564e4ca46dcdb199eabeec90ca
download 1QDKeAQ8Ka6_wxoN3Ld5t3EP9UHk7Fw98 val/iSAID_val.json \
  b0b6a43b6d42ba1bde43f21aa41c384598b820201ba7a2309b2f1ce4f4736822
download 1_PJy7kDVnp9tjETUbvwNjkLGpzDfqYh6 val/semantic_masks.zip \
  f3b333b4b0469edb1079c49c163c915ae8dd39d116191798633564bac1998d69
download 1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2 dota/train/part1.zip \
  8f2a06820798c3be88bbeef31a7b43e54ebe84adfb42c39085a87d9dea3514b7
download 1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v dota/train/part2.zip \
  18dd413b1d7ce9e1b5cfbbf18e524ebcc5be5064da7a4a8768a38b493a6c2905
download 1pEmwJtugIWhiwgBqOtplNUtTG2T454zn dota/train/part3.zip \
  5ff5ef2d492bbd55139cf3f39f839adeaa0fc8dfee0174dd7cb148402da0c823
download 1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP dota/val/part1.zip \
  e8f5602dc22c5142b3d2f26982c54df4e9e5e9b6b92015bc20a434e12b45ca9d

echo "done. Next: uv run python scripts/isaid_p0a.py"
