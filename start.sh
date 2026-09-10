#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -f "$ROOT/config.env" ]; then
    echo "ERROR: config.env is missing. Run ./setup.sh first." >&2
    exit 1
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "ERROR: Python environment is missing. Run ./setup.sh first." >&2
    exit 1
fi

if [ ! -f "$ROOT/src/main.py" ]; then
    echo "ERROR: src/main.py is missing." >&2
    exit 1
fi

set -a
source "$ROOT/config.env"
set +a

export JETSON_FRIEND_ROOT="$ROOT"

if [ -d /usr/local/cuda/bin ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

if [ -d /usr/local/cuda/lib64 ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

if [ -S "$XDG_RUNTIME_DIR/bus" ]; then
    export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
fi

if [ -z "${DISPLAY:-}" ]; then
    XSOCKET="$(find /tmp/.X11-unix -maxdepth 1 -type s -name 'X*' 2>/dev/null | sort -V | head -1 || true)"

    if [ -n "$XSOCKET" ]; then
        export DISPLAY=":${XSOCKET##*X}"
    fi
fi

if [ -z "${XAUTHORITY:-}" ] && [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
    export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
fi

exec "$ROOT/.venv/bin/python" "$ROOT/src/main.py" "$@"
