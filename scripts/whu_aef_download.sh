#!/usr/bin/env bash
# Download the AEF 2019 tiles covering the 100 WHU-OPT-SAR sheets (whole COGs).
# curl over the public HTTPS endpoint: gsutil's sliced download stalled at
# ~0.1 MB/s here while curl gets ~25 MB/s. Size is checked against
# Content-Length; a partial file resumes with --continue-at.
set -u
cd "${WHU_AEF_ROOT:-/home/yzhong/rs-eval-cache/whu-aef2019}"
mkdir -p tiles
while read -r url; do
  zone=$(basename "$(dirname "$url")"); name="${zone}_$(basename "$url")"
  https="https://storage.googleapis.com/${url#gs://}"
  want=$(curl -sfI "$https" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  have=$(stat -c %s "tiles/$name" 2>/dev/null || echo 0)
  if [ -n "$want" ] && [ "$have" = "$want" ]; then echo "skip $name"; continue; fi
  t0=$(date +%s)
  for attempt in 1 2 3 4 5; do
    curl -sfL --retry 3 --continue-at - -o "tiles/$name.part" "$https" && break
    sleep 5
  done
  got=$(stat -c %s "tiles/$name.part" 2>/dev/null || echo 0)
  if [ "$got" = "$want" ]; then mv "tiles/$name.part" "tiles/$name"; echo "ok $name $want $(( $(date +%s)-t0 ))s";
  else echo "FAIL $name got=$got want=$want"; fi
done < tiles.txt
echo DONE
