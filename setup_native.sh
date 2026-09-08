#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="$ROOT/deps"
MODELS="$ROOT/models"
VENV="$ROOT/.venv"

log() {
    printf '\n== %s ==\n' "$1"
}

ok() {
    printf 'OK: %s\n' "$1"
}

warn() {
    printf 'WARN: %s\n' "$1"
}

die() {
    printf 'ERROR: %s\n' "$1" >&2
    exit 1
}

download() {
    local url="$1"
    local target="$2"

    if [ -s "$target" ]; then
        ok "Already downloaded: $target"
    else
        mkdir -p "$(dirname "$target")"
        curl -L --fail --retry 5 --retry-delay 3 \
            -o "$target.part" "$url"
        mv "$target.part" "$target"
        ok "Downloaded: $target"
    fi
}

log "Jetson Friend native setup"

mkdir -p "$DEPS"
mkdir -p "$MODELS/llm"
mkdir -p "$MODELS/whisper"
mkdir -p "$MODELS/tts"
mkdir -p "$MODELS/vision"

log "Platform"

ARCH="$(uname -m)"
printf 'Architecture: %s\n' "$ARCH"

if [ "$ARCH" != "aarch64" ]; then
    warn "Expected aarch64 Jetson platform"
fi

if [ -r /proc/device-tree/model ]; then
    tr -d '\0' < /proc/device-tree/model
    printf '\n'
fi

if [ -r /etc/nv_tegra_release ]; then
    head -1 /etc/nv_tegra_release
fi

log "CUDA"

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    /usr/local/cuda/bin/nvcc --version
else
    die "CUDA compiler nvcc was not found"
fi

log "TensorRT"

if command -v trtexec >/dev/null 2>&1; then
    trtexec --help >/dev/null 2>&1 || true
    ok "TensorRT trtexec found"
else
    warn "trtexec not found. Vision TensorRT conversion will be skipped."
fi

log "System packages"

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    python3-opencv \
    python3-numpy \
    alsa-utils \
    pulseaudio-utils \
    pipewire-audio \
    v4l-utils \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    libopencv-dev \
    libopenblas-dev \
    espeak-ng

ok "System packages installed"

log "Python virtual environment"

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools

"$VENV/bin/python" -m pip install \
    numpy \
    sounddevice \
    soundfile \
    pygame \
    pyyaml \
    requests \
    psutil \
    pyudev \
    piper-tts

ok "Python environment ready"

log "Build llama.cpp with CUDA"

if [ ! -d "$DEPS/llama.cpp/.git" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp \
        "$DEPS/llama.cpp"
else
    git -C "$DEPS/llama.cpp" pull --ff-only
fi

rm -rf "$DEPS/llama.cpp/build"

cmake \
    -S "$DEPS/llama.cpp" \
    -B "$DEPS/llama.cpp/build" \
    -G Ninja \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cmake \
    --build "$DEPS/llama.cpp/build" \
    --config Release \
    -j"$(nproc)"

test -x "$DEPS/llama.cpp/build/bin/llama-cli" \
    || die "llama-cli was not built"

ok "llama.cpp CUDA build complete"

log "Build whisper.cpp with CUDA"

if [ ! -d "$DEPS/whisper.cpp/.git" ]; then
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp \
        "$DEPS/whisper.cpp"
else
    git -C "$DEPS/whisper.cpp" pull --ff-only
fi

rm -rf "$DEPS/whisper.cpp/build"

cmake \
    -S "$DEPS/whisper.cpp" \
    -B "$DEPS/whisper.cpp/build" \
    -G Ninja \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cmake \
    --build "$DEPS/whisper.cpp/build" \
    --config Release \
    -j"$(nproc)"

test -x "$DEPS/whisper.cpp/build/bin/whisper-cli" \
    || die "whisper-cli was not built"

ok "whisper.cpp CUDA build complete"

log "Download Qwen3-4B"

QWEN_URL="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=true"

download \
    "$QWEN_URL" \
    "$MODELS/llm/Qwen3-4B-Q4_K_M.gguf"

log "Download Whisper English model"

WHISPER_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin?download=true"

download \
    "$WHISPER_URL" \
    "$MODELS/whisper/ggml-base.en.bin"

log "Download Piper English voice"

PIPER_MODEL_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx?download=true"

PIPER_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx.json?download=true"

download \
    "$PIPER_MODEL_URL" \
    "$MODELS/tts/en_US-ryan-low.onnx"

download \
    "$PIPER_CONFIG_URL" \
    "$MODELS/tts/en_US-ryan-low.onnx.json"

log "Hardware discovery"

printf '\n--- Microphones ---\n'

if arecord -l; then
    ok "Audio input enumeration works"
else
    warn "No ALSA microphone detected"
fi

printf '\n--- Speakers ---\n'

if aplay -l; then
    ok "Audio output enumeration works"
else
    warn "No ALSA output detected"
fi

printf '\n--- Cameras ---\n'

CAMERAS="$(find /dev -maxdepth 1 -name 'video*' -print 2>/dev/null || true)"

if [ -n "$CAMERAS" ]; then
    printf '%s\n' "$CAMERAS"

    while IFS= read -r camera; do
        if [ -n "$camera" ]; then
            printf '\n%s\n' "$camera"
            v4l2-ctl --device="$camera" --all 2>/dev/null | head -20 || true
        fi
    done <<< "$CAMERAS"
else
    warn "No /dev/video* devices found"
fi

log "Smoke test: llama.cpp"

"$DEPS/llama.cpp/build/bin/llama-cli" \
    -m "$MODELS/llm/Qwen3-4B-Q4_K_M.gguf" \
    -ngl 99 \
    -c 512 \
    -n 16 \
    -p "Reply with exactly: Jetson Friend is alive." \
    || warn "LLM smoke test failed"

log "Piper"

if "$VENV/bin/python" -m piper --help >/dev/null 2>&1; then
    ok "Piper available"
else
    warn "Piper command test failed"
fi

log "Installation complete"

printf '\nNative Jetson Friend environment:\n'
printf 'LLM:     %s\n' "$MODELS/llm/Qwen3-4B-Q4_K_M.gguf"
printf 'Whisper: %s\n' "$MODELS/whisper/ggml-base.en.bin"
printf 'Piper:   %s\n' "$MODELS/tts/en_US-ryan-low.onnx"
printf 'Python:  %s\n' "$VENV/bin/python"

printf '\nNext step:\n'
printf 'Run ./start_native.sh after it is configured for the project.\n'
