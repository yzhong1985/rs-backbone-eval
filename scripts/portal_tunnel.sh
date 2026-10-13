#!/usr/bin/env bash
# Open a single-session SSH tunnel so THIS device's browser can reach the Portal.
#
# Run this ON THE DEVICE YOU SSH FROM (laptop / office machine / etc.):
#
#   scripts/portal_tunnel.sh [user@host] [local_port] [remote_port]
#
# Only one device may hold the Portal session: attaching here revokes the
# previous holder (its SSH session is closed, so its tunnel drops). The
# tunnel stays in the foreground; press Ctrl-C to detach.
#
# Environment overrides:
#   RS_EVAL_SSH_HOST       default user@host of the evaluation machine
#   RS_EVAL_REMOTE_REPO    repo path on the server, relative to $HOME
#                          (default: projects/rs-backbone-eval)
set -euo pipefail

SERVER="${1:-${RS_EVAL_SSH_HOST:-yzhong@192.168.1.89}}"
LOCAL_PORT="${2:-8080}"
REMOTE_PORT="${3:-8080}"
REMOTE_REPO="${RS_EVAL_REMOTE_REPO:-projects/rs-backbone-eval}"

REMOTE_CMD="cd \"\$HOME/${REMOTE_REPO}\" 2>/dev/null || cd \"${REMOTE_REPO}\"; exec bash scripts/portal_session.sh attach ${REMOTE_PORT}"

echo "Claiming the Portal session on ${SERVER} ..."
echo "Local:  http://127.0.0.1:${LOCAL_PORT}  ->  remote 127.0.0.1:${REMOTE_PORT}"
echo "Any previously connected device will be disconnected."
echo

exec ssh -t \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "$SERVER" "$REMOTE_CMD"
