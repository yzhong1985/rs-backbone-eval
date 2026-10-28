#!/usr/bin/env bash
# Manage the RS Backbone Eval Portal on this machine.
#
#   scripts/portal_server.sh start|stop|restart|status|url
#
# Environment overrides:
#   RS_EVAL_PORTAL_HOST  bind address (default 127.0.0.1; use 0.0.0.0 to expose on the LAN)
#   RS_EVAL_PORTAL_PORT  port (default 8080)
#   RS_EVAL_PROXY        HTTP proxy for downloads, e.g. http://127.0.0.1:1080
#                        (aria2c / gsutil / hf child processes inherit it)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fall back to the repo-local .env for values not present in the shell env.
env_value() {
    local key="$1"
    [ -f "$REPO_ROOT/.env" ] || return 0
    grep -E "^${key}=" "$REPO_ROOT/.env" | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'"
}

PORT="${RS_EVAL_PORTAL_PORT:-$(env_value RS_EVAL_PORTAL_PORT)}"
PORT="${PORT:-8080}"
BIND="${RS_EVAL_PORTAL_HOST:-$(env_value RS_EVAL_PORTAL_HOST)}"
BIND="${BIND:-127.0.0.1}"
PROXY="${RS_EVAL_PROXY:-$(env_value RS_EVAL_PROXY)}"
# Shared token for the public (frp) deployment: when set, every route except
# /api/health requires it (HTTP Basic, or ?key=<token> once to set a cookie).
TOKEN="${RS_EVAL_PORTAL_TOKEN:-$(env_value RS_EVAL_PORTAL_TOKEN)}"
export RS_EVAL_PORTAL_TOKEN="$TOKEN"
LOG="${RS_EVAL_PORTAL_LOG:-$HOME/.cache/rs-eval-portal.log}"
PIDFILE="${RS_EVAL_PORTAL_PIDFILE:-$HOME/.cache/rs-eval-portal.pid}"

export PATH="$HOME/.local/bin:$PATH"

health_url() { echo "http://127.0.0.1:${PORT}/api/health"; }

is_healthy() {
    curl -fsS --max-time 4 "$(health_url)" >/dev/null 2>&1
}

lan_ips() {
    hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192\.168|10\.|172\.(1[6-9]|2[0-9]|3[01]))' || true
}

print_access() {
    echo "Portal bind: ${BIND}:${PORT}"
    if [ "$BIND" = "0.0.0.0" ]; then
        for ip in $(lan_ips); do
            echo "  direct (LAN):   http://${ip}:${PORT}"
        done
    else
        echo "  local only:     http://127.0.0.1:${PORT}"
    fi
    if [ -n "${RS_EVAL_PORTAL_TOKEN:-}" ]; then
        echo "  auth: shared token required (Basic user is ignored, password = token)"
        echo "        or open \"?key=<token>\" once to get a cookie"
    else
        echo "  auth: OFF -- anyone who can reach this port can read every result"
    fi
    echo "From any remote device with SSH access to this host, either:"
    echo "  1) run on the remote device:"
    for ip in $(lan_ips | head -1); do
        echo "       scripts/portal_client.sh ${USER}@${ip} ${PORT}"
    done
    echo "  2) or, if ${BIND} is 0.0.0.0, open the direct LAN URL above."
}

cmd_start() {
    if is_healthy; then
        echo "Portal already healthy on port ${PORT}."
        print_access
        return 0
    fi
    mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"
    echo "Starting Portal on ${BIND}:${PORT} (log: ${LOG})"
    if [ -n "$PROXY" ]; then
        echo "Downloads will use proxy ${PROXY}"
        RS_EVAL_PORTAL_HOST="$BIND" RS_EVAL_PORTAL_PORT="$PORT" \
        http_proxy="$PROXY" https_proxy="$PROXY" \
            nohup uv run --project "$REPO_ROOT" rs-eval portal --host "$BIND" --port "$PORT" \
            >"$LOG" 2>&1 &
    else
        RS_EVAL_PORTAL_HOST="$BIND" RS_EVAL_PORTAL_PORT="$PORT" \
            nohup uv run --project "$REPO_ROOT" rs-eval portal --host "$BIND" --port "$PORT" \
            >"$LOG" 2>&1 &
    fi
    echo $! >"$PIDFILE"
    for _ in $(seq 1 20); do
        sleep 1
        is_healthy && break
    done
    if is_healthy; then
        echo "Portal is up (pid $(cat "$PIDFILE"))."
        print_access
    else
        echo "Portal did not become healthy; last log lines:"
        tail -n 15 "$LOG"
        return 1
    fi
}

cmd_stop() {
    local stopped=0
    if [ -f "$PIDFILE" ]; then
        local pid
        pid="$(cat "$PIDFILE")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && stopped=1
            echo "Stopped portal pid ${pid}."
        fi
        rm -f "$PIDFILE"
    fi
    pkill -f "rs-eval portal" 2>/dev/null && stopped=1
    [ "$stopped" = 0 ] && echo "No portal process found."
    return 0
}

cmd_status() {
    if is_healthy; then
        echo "healthy on port ${PORT}"
        ss -tlnp 2>/dev/null | grep ":${PORT}" || true
        print_access
    else
        echo "not running on port ${PORT}"
        return 1
    fi
}

case "${1:-status}" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    restart) cmd_stop; sleep 1; cmd_start ;;
    status) cmd_status ;;
    url) print_access ;;
    *)
        echo "usage: $0 start|stop|restart|status|url" >&2
        exit 2
        ;;
esac
