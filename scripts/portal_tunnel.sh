#!/usr/bin/env bash
# Open a local SSH tunnel to the rs-eval Portal on the evaluation host.
#
# Usage:
#   ./portal_tunnel.sh <user@host> <ssh-port> [local-port] [remote-port] [remote-repo]
#
# Examples:
#   ./portal_tunnel.sh yzhong@domain.example 23456
#   ./portal_tunnel.sh yzhong@domain.example 23456 8081
#   ./portal_tunnel.sh --host yzhong@domain.example --ssh-port 23456 --local-port 8081
#
# Flags may be placed before the positional arguments:
#   --key PATH          SSH private key (default: ssh-agent/default config)
#   --repo PATH         remote repo relative to $HOME or absolute
#                       (default: projects/rs-backbone-eval)
#   --no-browser        do not try to open the URL
#   --dry-run           print the ssh command only
#
# The remote host must be able to reach the Portal on 127.0.0.1:8080.
# If the remote portal_server_session.sh exists, the tunnel claims the
# single Portal session (new clients revoke the previous one); otherwise it
# just holds a plain tunnel.
set -euo pipefail

HOST=""
SSH_PORT=""
LOCAL_PORT=8080
REMOTE_PORT=8080
REMOTE_REPO="projects/rs-backbone-eval"
KEY=""
OPEN_BROWSER=1
DRY_RUN=0

usage() {
    cat <<'EOF'
Open a local SSH tunnel to the rs-eval Portal on the evaluation host.

Usage:
  ./portal_tunnel.sh <user@host> <ssh-port> [local-port] [remote-port] [remote-repo]

Examples:
  ./portal_tunnel.sh yzhong@domain.example 23456
  ./portal_tunnel.sh yzhong@domain.example 23456 8081
  ./portal_tunnel.sh --host yzhong@domain.example --ssh-port 23456 --local-port 8081

Flags must come before the positional arguments:
  --key PATH          SSH private key (default: ssh-agent/default config)
  --repo PATH         remote repo relative to $HOME or absolute
                      (default: projects/rs-backbone-eval)
  --no-browser        do not try to open the URL
  --dry-run           print the ssh command only

The remote host must be able to reach the Portal on 127.0.0.1:8080.
If the remote portal_server_session.sh exists, the tunnel claims the
single Portal session (new clients revoke the previous one); otherwise it
just holds a plain tunnel.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

need_value() {
    [ "$#" -ge 2 ] && [ -n "$2" ] || die "missing value for $1"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --host) need_value "$1" "${2:-}"; HOST="$2"; shift 2 ;;
        --ssh-port) need_value "$1" "${2:-}"; SSH_PORT="$2"; shift 2 ;;
        --local-port) need_value "$1" "${2:-}"; LOCAL_PORT="$2"; shift 2 ;;
        --remote-port) need_value "$1" "${2:-}"; REMOTE_PORT="$2"; shift 2 ;;
        --repo) need_value "$1" "${2:-}"; REMOTE_REPO="$2"; shift 2 ;;
        --key) need_value "$1" "${2:-}"; KEY="$2"; shift 2 ;;
        --no-browser) OPEN_BROWSER=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*) die "unknown option: $1 (try --help)" ;;
        *) break ;;
    esac
done

# Positional fallback: host ssh-port [local-port] [remote-port] [repo]
if [ -z "$HOST" ]; then
    [ "$#" -ge 2 ] || { usage; exit 2; }
    HOST="$1"; SSH_PORT="$2"; shift 2
    [ "$#" -ge 1 ] && { LOCAL_PORT="$1"; shift; }
    [ "$#" -ge 1 ] && { REMOTE_PORT="$1"; shift; }
    [ "$#" -ge 1 ] && { REMOTE_REPO="$1"; shift; }
fi

[ -n "$HOST" ] || { usage; exit 2; }
[ -n "$SSH_PORT" ] || die "missing SSH port (frp port)"
[[ "$SSH_PORT" =~ ^[0-9]+$ ]] || die "SSH port must be numeric: $SSH_PORT"
[[ "$LOCAL_PORT" =~ ^[0-9]+$ ]] || die "local port must be numeric: $LOCAL_PORT"
[[ "$REMOTE_PORT" =~ ^[0-9]+$ ]] || die "remote port must be numeric: $REMOTE_PORT"
[[ "$HOST" == *@* ]] || die "host must be user@host, got: $HOST"

if [ -n "$KEY" ]; then
    KEY="${KEY/#\~/$HOME}"
    [ -f "$KEY" ] || die "SSH key not found: $KEY"
fi

port_in_use() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

# If the default local port is busy, pick the next free one. An explicitly
# supplied port is also auto-bumped rather than failing the whole run.
if port_in_use "$LOCAL_PORT"; then
    original="$LOCAL_PORT"
    found=0
    for candidate in {8081..8090}; do
        if [ "$candidate" != "$original" ] && ! port_in_use "$candidate"; then
            LOCAL_PORT="$candidate"
            found=1
            break
        fi
    done
    [ "$found" = 1 ] || die "local port $original is busy and 8081-8090 are also busy"
    echo "local port $original busy; using $LOCAL_PORT instead"
fi

URL="http://127.0.0.1:${LOCAL_PORT}"

if [[ "$REMOTE_REPO" = /* ]]; then
    REPO_CD="cd $(printf '%q' "$REMOTE_REPO")"
else
    REPO_CD="cd \"\$HOME\"/$(printf '%q' "$REMOTE_REPO")"
fi

REMOTE_CMD="$REPO_CD 2>/dev/null || { echo 'remote repo not found: $REMOTE_REPO' >&2; exit 1; }; \
if [ -f scripts/portal_server_session.sh ]; then \
    exec bash scripts/portal_server_session.sh attach $REMOTE_PORT; \
else \
    echo 'portal_server_session.sh not found; holding a plain tunnel' >&2; \
    exec sleep 2147483647; \
fi"

SSH_ARGS=(
    -p "$SSH_PORT"
    -t
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"
)
if [ -n "$KEY" ]; then
    SSH_ARGS+=(-i "$KEY" -o IdentitiesOnly=yes)
fi

echo "Portal:  $URL"
echo "Tunnel:  127.0.0.1:${LOCAL_PORT} -> ${HOST}:${REMOTE_PORT} (SSH port ${SSH_PORT})"
echo "Repo:    ${REMOTE_REPO}"
echo "Ctrl-C to close."

if [ "$DRY_RUN" = 1 ]; then
    printf 'ssh'
    printf ' %q' "${SSH_ARGS[@]}"
    printf ' %q' "$HOST" "$REMOTE_CMD"
    printf '\n'
    exit 0
fi

open_url() {
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$1" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
        open "$1" >/dev/null 2>&1 || true
    elif command -v cmd.exe >/dev/null 2>&1; then
        cmd.exe /c start "" "$1" >/dev/null 2>&1 || true
    else
        echo "open this in your browser: $1"
    fi
}

if [ "$OPEN_BROWSER" = 1 ]; then
    ( sleep 2; open_url "$URL" ) &
fi

exec ssh "${SSH_ARGS[@]}" "$HOST" "$REMOTE_CMD"
