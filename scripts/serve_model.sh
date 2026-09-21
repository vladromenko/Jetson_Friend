#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set -a; source "$ROOT/config.env"; set +a
exec "$LLAMA_SERVER_BIN" -m "$LLM_MODEL" --mmproj "$VLM_MMPROJ" -c "$LLM_CTX" -ngl "$LLM_GPU_LAYERS" -ub "$LLM_UBATCH" --host "$LLM_HOST" --port "$LLM_PORT"
