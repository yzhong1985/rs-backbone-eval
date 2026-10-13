#!/usr/bin/env bash
# Server-side single-session gate for Portal tunnels (run on the evaluation
# machine; normally invoked remotely by scripts/portal_client.sh).
#
# Only one remote device may hold a Portal tunnel at a time: when a new
# client attaches, the previous holder is revoked, which terminates its SSH
# session and closes its tunnel.
#
#   scripts/portal_server_session.sh attach [port]   # claim the session, hold it open
#   scripts/portal_server_session.sh status
#   scripts/portal_server_session.sh release
#
# Normally invoked by scripts/portal_client.sh from the remote device.
set -uo pipefail

STATE_DIR="${RS_EVAL_SESSION_DIR:-$HOME/.cache/rs-eval}"
LOCK_FILE="${STATE_DIR}/portal-session.pid"
LOCKFILE="${STATE_DIR}/portal-session.lock"
mkdir -p "$STATE_DIR"

_revoke_pid() {
    local pid="$1"
    [ -n "$pid" ] || return 0
    [ "$pid" = "$$" ] && return 0
    kill -0 "$pid" 2>/dev/null || return 0
    echo "Revoking previous Portal session (pid ${pid})."
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "$pid" 2>/dev/null || true
    return 0
}

cmd_attach() {
    local port="${1:-8080}" previous=""
    mkdir -p "$STATE_DIR"
    exec 9>"$LOCKFILE"
    flock 9
    [ -f "$LOCK_FILE" ] && previous="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    echo "$$" >"$LOCK_FILE"          # claim first; the victim must not delete our record
    flock -u 9

    _revoke_pid "$previous"          # then revoke exactly the recorded predecessor

    _cleanup() {
        # Only remove the record if it still belongs to this session.
        if [ -f "$LOCK_FILE" ] && [ "$(cat "$LOCK_FILE" 2>/dev/null)" = "$$" ]; then
            rm -f "$LOCK_FILE" 2>/dev/null
        fi
        echo
        echo "Portal session released."
    }
    trap '_cleanup' EXIT
    trap 'exit 0' INT TERM HUP
    echo "Portal session active on this device (pid $$, remote port ${port})."
    echo "http://127.0.0.1:${port}  <- open this on your local machine"
    echo "Ctrl-C here (or close your SSH connection) detaches."
    # Hold the session open with an interruptible wait so SIGTERM is handled
    # immediately (a foreground `sleep` would defer the trap until it ended).
    while :; do
        sleep 3600 &
        wait $! || break
    done
}

cmd_status() {
    if [ ! -f "$LOCK_FILE" ]; then
        echo "no active Portal session"
        return 1
    fi
    local pid
    pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "active Portal session: pid ${pid} ($(ps -o lstart= -p "$pid" 2>/dev/null | xargs))"
    else
        echo "stale session record (pid ${pid}); the holder is gone"
        return 1
    fi
}

cmd_release() {
    _revoke_pid "$(cat "$LOCK_FILE" 2>/dev/null || true)"
    rm -f "$LOCK_FILE"
    echo "released"
}

case "${1:-status}" in
    attach) cmd_attach "${2:-8080}" ;;
    status) cmd_status ;;
    release) cmd_release ;;
    *)
        echo "usage: $0 attach [port] | status | release" >&2
        exit 2
        ;;
esac
