#!/usr/bin/env bash
# Open an SSH tunnel so this device's browser can reach the Portal.
#
# Run this ON THE DEVICE YOU SSH FROM (your laptop / another machine):
#
#   scripts/portal_tunnel.sh [user@host] [local_port] [remote_port]
#
# Defaults: host from RS_EVAL_SSH_HOST or yzhong@<this-repo-host-ip>,
# local port 8080 -> remote 8080. Then open http://127.0.0.1:<local_port>.
#
# The tunnel stays in the foreground; press Ctrl-C to close it.
set -euo pipefail

SERVER="${1:-${RS_EVAL_SSH_HOST:-yzhong@192.168.1.89}}"
LOCAL_PORT="${2:-8080}"
REMOTE_PORT="${3:-8080}"

echo "Tunnelling 127.0.0.1:${LOCAL_PORT} -> ${SERVER} (remote 127.0.0.1:${REMOTE_PORT})"
echo "Open:  http://127.0.0.1:${LOCAL_PORT}"
echo "Press Ctrl-C to close the tunnel."
echo

exec ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "$SERVER"
