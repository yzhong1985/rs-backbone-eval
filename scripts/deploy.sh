#!/usr/bin/env bash
# Deploy one pinned commit of this repo to a directory for running experiments.
#
#   bash scripts/deploy.sh --target DIR --ref REF [--source URL|BUNDLE] [--skip-sync] [--dry-run] [--allow-busy]
#
# * TARGET becomes a clean, detached checkout at REF. Do not edit it by hand.
# * Every environment (the root one and each workers/<name>) is built inside TARGET
#   with uv sync. Virtualenvs embed absolute paths, so they cannot be copied.
# * --source defaults to this repo's origin. A git bundle path also works, for hosts
#   that cannot reach GitHub (AGENTS.md: the 3080 host syncs by bundle).
# * A deploy record goes to DIR.deploy.json, outside the checkout, so the tree stays clean.
# * Refuses to run while the queue has running or pending jobs (they execute from the
#   deploy checkout). --allow-busy overrides this; RS_EVAL_QUEUE_ROOT sets the queue dir.
# * .env is not created here. Copy .env.example to TARGET/.env and fill the paths,
#   then run `bash scripts/preflight.sh` inside TARGET (this script runs it at the end).
set -euo pipefail

die() { echo "deploy: $*" >&2; exit 1; }
usage() { sed -n '2,15p' "$0"; exit 2; }

TARGET=""; REF=""; SOURCE=""; SKIP_SYNC=0; DRY=0; ALLOW_BUSY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --ref) REF="${2:-}"; shift 2 ;;
    --source) SOURCE="${2:-}"; shift 2 ;;
    --skip-sync) SKIP_SYNC=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --allow-busy) ALLOW_BUSY=1; shift ;;
    -h|--help) usage ;;
    *) echo "deploy: unknown argument $1" >&2; usage ;;
  esac
done
[[ -n "$TARGET" && -n "$REF" ]] || usage

HERE="$(cd "$(dirname "$0")/.." && pwd)"
[[ -n "$SOURCE" ]] || SOURCE="$(git -C "$HERE" remote get-url origin)"
RECORD="${TARGET%/}.deploy.json"

if [[ $DRY -eq 1 ]]; then
  echo "plan: source=$SOURCE ref=$REF target=$TARGET record=$RECORD skip_sync=$SKIP_SYNC"
  exit 0
fi

# 0. the queue runs jobs from the deploy checkout: do not change it under a running job.
# The queue dir comes from the environment, else from the target's .env (the queue the
# target will actually serve), else the default. The check runs under deploy.lock,
# which the queue consumer also takes when it claims a job, so a job cannot start
# between the check and the checkout change.
QUEUE_ROOT="${RS_EVAL_QUEUE_ROOT:-}"
if [[ -z "$QUEUE_ROOT" && -f "$TARGET/.env" ]]; then
  QUEUE_ROOT="$(sed -n 's/^RS_EVAL_QUEUE_ROOT=//p' "$TARGET/.env" | tail -1)"
fi
QUEUE_ROOT="${QUEUE_ROOT:-/geodata/rs-eval/queue}"
if [[ $ALLOW_BUSY -eq 0 ]]; then
  mkdir -p "$QUEUE_ROOT"
  exec 9>"$QUEUE_ROOT/deploy.lock"
  flock 9
  busy=$(ls "$QUEUE_ROOT/running" "$QUEUE_ROOT/pending" 2>/dev/null | grep -c '\.json$' || true)
  [[ "$busy" -eq 0 ]] || die "queue has $busy running/pending job(s) in $QUEUE_ROOT; wait, or pass --allow-busy"
fi

# 1. checkout: clone once, then fetch; refuse a dirty target.
if [[ ! -d "$TARGET/.git" ]]; then
  git clone -q --no-checkout "$SOURCE" "$TARGET"
else
  [[ -z "$(git -C "$TARGET" status --porcelain)" ]] || die "$TARGET has local changes; refusing to deploy"
fi
# --prune drops deploy-src refs whose branch was deleted at SOURCE. Without it, a
# removed branch keeps resolving to its last commit and that commit gets deployed.
git -C "$TARGET" fetch -q --prune "$SOURCE" "+refs/heads/*:refs/remotes/deploy-src/*" \
  || die "fetch from $SOURCE failed"
git -C "$TARGET" fetch -q --prune "$SOURCE" "+refs/tags/*:refs/tags/*" \
  || die "fetch of tags from $SOURCE failed"
# REF is resolved against the fetched source only. Local branches in TARGET are ignored,
# so a stale local "main" cannot shadow origin's "main". Commit hashes are the last resort.
resolve_commit() {
  local ref="$1" name commit
  for name in "refs/remotes/deploy-src/$ref" "refs/tags/$ref"; do
    if commit="$(git -C "$TARGET" rev-parse --verify --quiet "$name^{commit}")"; then
      echo "$commit"
      return 0
    fi
  done
  if [[ "$ref" =~ ^[0-9a-fA-F]{7,40}$ ]] \
    && commit="$(git -C "$TARGET" rev-parse --verify --quiet "$ref^{commit}")"; then
    echo "$commit"
    return 0
  fi
  return 1
}
COMMIT="$(resolve_commit "$REF")" \
  || die "ref $REF not found in $SOURCE (looked in branches, tags and commit hashes)"
git -C "$TARGET" checkout -q --detach "$COMMIT"
echo "checked out $COMMIT in $TARGET"

# 2. environments: the root one first, then every worker, as the Dockerfile does.
if [[ $SKIP_SYNC -eq 0 ]]; then
  export UV_LINK_MODE=hardlink
  (cd "$TARGET" && uv sync --frozen --extra portal --extra downloads --extra dev)
  for worker in "$TARGET"/workers/*/; do
    [[ -f "$worker/pyproject.toml" ]] || continue
    (cd "$TARGET" && uv sync --frozen --project "workers/$(basename "$worker")")
  done
fi

# 3. record, outside the checkout.
printf '{\n  "commit": "%s",\n  "ref": "%s",\n  "source": "%s",\n  "target": "%s",\n  "host": "%s",\n  "deployed_at": "%s"\n}\n' \
  "$COMMIT" "$REF" "$SOURCE" "$TARGET" "$(hostname)" "$(date -Iseconds)" > "$RECORD"
echo "wrote $RECORD"

# 4. readiness. A missing .env is reported, not created.
if [[ ! -f "$TARGET/.env" ]]; then
  echo "next: cp $TARGET/.env.example $TARGET/.env and fill the data, weights, runs and cache paths"
fi
[[ -f "$TARGET/scripts/preflight.sh" ]] && (cd "$TARGET" && bash scripts/preflight.sh) || true
