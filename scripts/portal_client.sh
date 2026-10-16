#!/usr/bin/env bash
# Client-side helper: open a single-session SSH tunnel so THIS device's
# browser can reach the Portal (run it on the device you SSH from).
#
# Run this ON THE DEVICE YOU SSH FROM (laptop / office machine / etc.):
#
#   scripts/portal_client.sh [user@host] [local_port] [remote_port]
#
# Only one device may hold the Portal session: attaching here revokes the
# previous holder (its SSH session is closed, so its tunnel drops). The
# tunnel stays in the foreground; press Ctrl-C to detach.
#
# Environment overrides:
#   RS_EVAL_SSH_HOST       default user@host of the evaluation machine
#   RS_EVAL_SSH_PORT       ssh port when the host is reached through an
#                          frps/relay mapping whose public side is not 22
#   RS_EVAL_SSH_EXTRA      extra ssh flags, e.g. "-i ~/keys/gpu.pem -o IdentitiesOnly=yes"
#   RS_EVAL_REMOTE_REPO    repo path on the server, relative to $HOME
#                          (default: projects/rs-backbone-eval)
set -euo pipefail

SERVER="${1:-${RS_EVAL_SSH_HOST:-yzhong@192.168.1.89}}"
LOCAL_PORT="${2:-8080}"
REMOTE_PORT="${3:-8080}"
REMOTE_REPO="${RS_EVAL_REMOTE_REPO:-projects/rs-backbone-eval}"

# Tail a swagger of extra ssh flags before the destination token, so an
# frp/relay setup works unchanged from the outside: the port that its frps
# maps to the evaluation machine is not 22 at the exit node.
PORT_ARGS=()
if [ -n "${RS_EVAL_SSH_PORT:-}" ]; then
    PORT_ARGS+=(-p "${RS_EVAL_SSH_PORT}")
fi
while IFS= read -r flag; do
    [ -n "$flag" ] && PORT_ARGS+=("$flag")
done <<<"${RS_EVAL_SSH_EXTRA:-}"

REMOTE_CMD="cd \"\$HOME/${REMOTE_REPO}\" 2>/dev/null || cd \"${REMOTE_REPO}\"; exec bash scripts/portal_server_session.sh attach ${REMOTE_PORT}"

echo "Claiming the Portal session on ${SERVER} ..."
echo "Local:  http://127.0.0.1:${LOCAL_PORT}  ->  remote 127.0.0.1:${REMOTE_PORT}"
echo "Any previously connected device will be disconnected."
echo

exec ssh -t \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    "${PORT_ARGS[@]}" \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "$SERVER" "$REMOTE_CMD"
