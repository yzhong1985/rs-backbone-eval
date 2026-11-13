#!/usr/bin/env bash
# Ask the codex agent on the WSL review host to review a commit in a read-only
# sandbox, over a reverse SSH tunnel from that host back to this machine.
#
# Why a tunnel: the WSL host sits behind NAT (and, in this deployment, behind
# frpc), so this machine cannot reach its sshd directly. The review host instead
# opens `ssh -R 2222:localhost:22 user@<this-host>`; this script connects to
# 127.0.0.1:2222.
#
# Safety: the review runs in a dedicated clone (~/review/rs-backbone-eval), not
# the operator's working tree, with `codex exec -s read-only review`. The agent
# never sees /geodata/rs-eval/runs, datasets or an experimental deployment
# checkout, and cannot write files. A sibling codex session once deleted a
# deployment checkout; this is the guard against a repeat.
#
# Usage:
#   scripts/remote_review.sh              # review HEAD
#   scripts/remote_review.sh <sha>        # review a specific commit
#
# Environment overrides:
#   RS_EVAL_REVIEW_PORT   tunnel port           (default 2222)
#   RS_EVAL_REVIEW_HOST   tunnel entry host     (default 127.0.0.1)
#   RS_EVAL_REVIEW_USER   review host login     (default yzhong)
#   RS_EVAL_REVIEW_REPO   clone on review host  (default ~/review/rs-backbone-eval)
#   RS_EVAL_REVIEW_NODE   node bin dir on host  (default ~/.nvm/versions/node/v22.23.2/bin)
#   RS_EVAL_REVIEW_PROXY  proxy for chatgpt.com (default http://<wsl-gateway>:10808)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TARGET="${1:-HEAD}"
if ! SHA="$(git -C "$REPO_ROOT" rev-parse --verify "${TARGET}^{commit}" 2>/dev/null)"; then
    echo "error: unknown commit: $TARGET" >&2
    exit 2
fi
SHORT="$(git -C "$REPO_ROOT" rev-parse --short "$SHA")"

PORT="${RS_EVAL_REVIEW_PORT:-2222}"
HOST="${RS_EVAL_REVIEW_HOST:-127.0.0.1}"
USER="${RS_EVAL_REVIEW_USER:-yzhong}"
REMOTE_REPO="${RS_EVAL_REVIEW_REPO:-}"
REMOTE_NODE="${RS_EVAL_REVIEW_NODE:-}"
PROXY="${RS_EVAL_REVIEW_PROXY:-}"

echo "reviewing $SHORT via $USER@$HOST:$PORT" >&2

ssh -p "$PORT" -o BatchMode=yes -o ConnectTimeout=10 "$USER@$HOST" \
    bash -s -- "$SHA" "$SHORT" "$REMOTE_REPO" "$REMOTE_NODE" "$PROXY" <<'REMOTE'
set -euo pipefail
sha="$1"
short="$2"
repo="${3:-$HOME/review/rs-backbone-eval}"
node_bin="${4:-$HOME/.nvm/versions/node/v22.23.2/bin}"
proxy="${5:-}"

# Non-interactive ssh does not load .bashrc, so replicate the pieces codex
# needs: the proxy to chatgpt.com and the nvm-provided node/codex binaries.
# Never source .bashrc here: it also exports unrelated API keys.
if [ -n "$proxy" ]; then
    http_proxy="$proxy"
    https_proxy="$proxy"
else
    wsl_host="$(ip route | awk '/^default/{print $3; exit}')"
    http_proxy="http://${wsl_host}:10808"
    https_proxy="$http_proxy"
fi
export http_proxy https_proxy
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export PATH="${node_bin}:$PATH"

cd "$repo"
git fetch --prune --quiet origin
if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
    echo "error: $sha not found after fetch; did the push land?" >&2
    exit 3
fi
git checkout --detach --quiet "$sha"
git clean -ffd --quiet

mkdir -p "$HOME/review"
out="$HOME/review/${short}.md"
echo "codex sandbox: read-only; output: $out" >&2
codex exec -s read-only review --commit "$sha" 2>&1 | tee "$out"
REMOTE
