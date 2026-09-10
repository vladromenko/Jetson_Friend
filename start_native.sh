#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"

if [ ! -f "$ROOT/config.env" ]; then
    echo "Missing config.env"
    exit 1
fi

set -a
source "$ROOT/config.env"
set +a

XSOCKET=$(find /tmp/.X11-unix -maxdepth 1 -type s -name 'X*' 2>/dev/null | sort -V | head -1)
if [ -n "$XSOCKET" ]; then
    export DISPLAY=":${XSOCKET##*X}"
fi

export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"

export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "Python virtual environment not found."
    exit 1
fi

exec "$ROOT/.venv/bin/python" "$ROOT/src/main.py" "$@"
