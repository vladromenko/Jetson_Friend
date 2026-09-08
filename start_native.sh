#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -f "$ROOT/config.env" ]; then
    set -a
    source "$ROOT/config.env"
    set +a
fi

if [ -f "$ROOT/.venv/bin/activate" ]; then
    source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DISPLAY="${DISPLAY:-:1}"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$ROOT/deps/llama.cpp/build/bin/llama-server}"
LLAMA_SERVER_HOST="${LLAMA_SERVER_HOST:-127.0.0.1}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8081}"
LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://${LLAMA_SERVER_HOST}:${LLAMA_SERVER_PORT}/v1/chat/completions}"
export LLAMA_SERVER_URL

SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

if [ ! -x "$LLAMA_SERVER_BIN" ]; then
    echo "ERROR: llama-server not found: $LLAMA_SERVER_BIN"
    exit 1
fi

if [ ! -f "$LLM_MODEL" ]; then
    echo "ERROR: VLM model not found: $LLM_MODEL"
    exit 1
fi

if [ ! -f "$VLM_MMPROJ" ]; then
    echo "ERROR: VLM projector not found: $VLM_MMPROJ"
    exit 1
fi

if ! curl -fsS "http://${LLAMA_SERVER_HOST}:${LLAMA_SERVER_PORT}/health" >/dev/null 2>&1; then
    echo "Starting persistent Hugh VLM server..."

    "$LLAMA_SERVER_BIN" \
        -m "$LLM_MODEL" \
        --mmproj "$VLM_MMPROJ" \
        -c "${LLAMA_CTX:-4096}" \
        -np 1 \
        -ngl "${LLAMA_GPU_LAYERS:-99}" \
        --fit-target 2048 \
        --image-min-tokens "${VLM_IMAGE_MIN_TOKENS:-1024}" \
        --host "$LLAMA_SERVER_HOST" \
        --port "$LLAMA_SERVER_PORT" \
        > /tmp/hugh-llama-server.log 2>&1 &

    SERVER_PID=$!

    READY=0

    for i in $(seq 1 120); do
        if curl -fsS "http://${LLAMA_SERVER_HOST}:${LLAMA_SERVER_PORT}/health" >/dev/null 2>&1; then
            READY=1
            break
        fi

        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: llama-server stopped during startup."
            tail -80 /tmp/hugh-llama-server.log || true
            exit 1
        fi

        sleep 0.5
    done

    if [ "$READY" -ne 1 ]; then
        echo "ERROR: llama-server did not become ready."
        tail -80 /tmp/hugh-llama-server.log || true
        exit 1
    fi
fi

exec "$ROOT/.venv/bin/python" "$ROOT/src/main.py" "$@"
