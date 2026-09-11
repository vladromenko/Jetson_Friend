#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="$ROOT/deps"
MODELS="$ROOT/models"
VENV="$ROOT/.venv"
ENV_FILE="$ROOT/config.env"

WITH_VLM=0
SKIP_BUILD=0
SKIP_MODELS=0

for arg in "$@"; do
    case "$arg" in
        --with-vlm)
            WITH_VLM=1
            ;;
        --skip-build)
            SKIP_BUILD=1
            ;;
        --skip-models)
            SKIP_MODELS=1
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: ./setup.sh [--with-vlm] [--skip-build] [--skip-models]" >&2
            exit 2
            ;;
    esac
done

log() {
    printf '\n============================================================\n'
    printf '== %s\n' "$*"
    printf '============================================================\n'
}

ok() {
    printf 'OK: %s\n' "$*"
}

warn() {
    printf 'WARN: %s\n' "$*" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_jetson() {
    local arch
    arch="$(uname -m)"

    if [ "$arch" != "aarch64" ]; then
        die "This installer must be run on the Jetson, not on the Mac."
    fi

    if [ -r /proc/device-tree/model ]; then
        local model
        model="$(tr -d '\0' < /proc/device-tree/model)"
        printf 'Device: %s\n' "$model"

        if [[ "$model" != *Jetson* ]]; then
            warn "ARM64 detected, but device name does not contain Jetson."
        fi
    fi
}

download_file() {
    local url="$1"
    local target="$2"

    mkdir -p "$(dirname "$target")"

    if [ -s "$target" ]; then
        ok "$(basename "$target") already exists"
        return
    fi

    rm -f "$target.part"

    curl \
        -L \
        --fail \
        --retry 5 \
        --retry-delay 3 \
        --connect-timeout 30 \
        --progress-bar \
        -o "$target.part" \
        "$url"

    if [ ! -s "$target.part" ]; then
        rm -f "$target.part"
        die "Downloaded file is empty: $target"
    fi

    mv "$target.part" "$target"
    ok "Downloaded $(basename "$target")"
}

log "MILO unified installation"
require_jetson

mkdir -p \
    "$DEPS" \
    "$MODELS/llm" \
    "$MODELS/vlm" \
    "$MODELS/whisper" \
    "$MODELS/tts" \
    "$MODELS/vision" \
    "$ROOT/data" \
    "$ROOT/data/people" \
    "$ROOT/data/memory"

log "NVIDIA stack"

if [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

command -v nvcc >/dev/null 2>&1 || die "CUDA nvcc not found"
nvcc --version | tail -4

if command -v trtexec >/dev/null 2>&1; then
    ok "TensorRT trtexec found"
else
    warn "TensorRT trtexec not found. ONNX fallback can still be used."
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
    ccache \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-numpy \
    python3-yaml \
    python3-gi \
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
    libomp-dev \
    libgl1 \
    libglib2.0-0 \
    libsdl2-2.0-0 \
    espeak-ng \
    jq

ok "System packages installed"

log "Python environment"

if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi

if [ -f "$VENV/pyvenv.cfg" ]; then
    sed -i \
        's/^include-system-site-packages = false/include-system-site-packages = true/' \
        "$VENV/pyvenv.cfg"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

"$VENV/bin/python" -m pip install \
    sounddevice \
    soundfile \
    pygame \
    pyyaml \
    requests \
    psutil \
    pyudev \
    piper-tts \
    huggingface_hub

if "$VENV/bin/python" -m pip show numpy >/dev/null 2>&1; then
    NUMPY_LOCATION="$("$VENV/bin/python" -m pip show numpy | awk -F': ' '/^Location:/ {print $2}')"

    if [[ "$NUMPY_LOCATION" == "$VENV"* ]]; then
        warn "Removing pip NumPy so OpenCV can use Jetson/Ubuntu NumPy."
        "$VENV/bin/python" -m pip uninstall -y numpy
    fi
fi

"$VENV/bin/python" - <<'PY'
import cv2
import numpy
import pygame
import psutil
import pyudev
import requests
import sounddevice
import soundfile
import yaml

print("Python imports: OK")
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
PY

if [ "$SKIP_BUILD" -eq 0 ]; then
    log "Build llama.cpp with CUDA"

    if [ ! -d "$DEPS/llama.cpp/.git" ]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$DEPS/llama.cpp"
    else
        git -C "$DEPS/llama.cpp" pull --ff-only
    fi

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

    test -x "$DEPS/llama.cpp/build/bin/llama-server" || die "llama-server was not built"
    test -x "$DEPS/llama.cpp/build/bin/llama-cli" || die "llama-cli was not built"
    ok "llama.cpp ready"

    log "Build whisper.cpp with CUDA"

    if [ ! -d "$DEPS/whisper.cpp/.git" ]; then
        git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$DEPS/whisper.cpp"
    else
        git -C "$DEPS/whisper.cpp" pull --ff-only
    fi

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

    test -x "$DEPS/whisper.cpp/build/bin/whisper-cli" || die "whisper-cli was not built"
    ok "whisper.cpp ready"
else
    ok "Native builds skipped"
fi

log "Core speech and vision models"

download_file \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin?download=true" \
    "$MODELS/whisper/ggml-base.en.bin"

download_file \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx?download=true" \
    "$MODELS/tts/en_US-ryan-low.onnx"

download_file \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx.json?download=true" \
    "$MODELS/tts/en_US-ryan-low.onnx.json"

download_file \
    "https://raw.githubusercontent.com/yoobright/yolo-onnx/master/yolov8n.onnx" \
    "$MODELS/vision/yolov8n.onnx"

download_file \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    "$MODELS/vision/face_detection_yunet.onnx"

download_file \
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?download=true" \
    "$MODELS/vision/face_recognition_sface.onnx"

download_file \
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml" \
    "$MODELS/vision/coco.yaml"

if [ "$SKIP_MODELS" -eq 0 ]; then
    log "Default language model"

    "$VENV/bin/python" "$ROOT/src/model_downloader.py" download qwen3-4b-q4

    if [ "$WITH_VLM" -eq 1 ]; then
        log "Optional vision-language model"
        "$VENV/bin/python" "$ROOT/src/model_downloader.py" download qwen2.5-vl-3b-q4
    fi
else
    ok "LLM/VLM downloads skipped"
fi

log "TensorRT vision engines"

if command -v trtexec >/dev/null 2>&1; then
    if [ ! -s "$MODELS/vision/yolov8n.engine" ]; then
        if trtexec \
            --onnx="$MODELS/vision/yolov8n.onnx" \
            --saveEngine="$MODELS/vision/yolov8n.engine" \
            --fp16 \
            --memPoolSize=workspace:1024 \
            --skipInference \
            >"$ROOT/trt_yolo.log" 2>&1
        then
            ok "YOLO TensorRT engine created"
        else
            rm -f "$MODELS/vision/yolov8n.engine"
            warn "YOLO TensorRT conversion failed; ONNX fallback remains available."
        fi
    else
        ok "YOLO TensorRT engine already exists"
    fi

    if [ ! -s "$MODELS/vision/face_detection_yunet.engine" ]; then
        if trtexec \
            --onnx="$MODELS/vision/face_detection_yunet.onnx" \
            --saveEngine="$MODELS/vision/face_detection_yunet.engine" \
            --fp16 \
            --memPoolSize=workspace:512 \
            --skipInference \
            >"$ROOT/trt_face.log" 2>&1
        then
            ok "YuNet TensorRT engine created"
        else
            rm -f "$MODELS/vision/face_detection_yunet.engine"
            warn "YuNet TensorRT conversion failed; ONNX fallback remains available."
        fi
    else
        ok "YuNet TensorRT engine already exists"
    fi
fi

log "Configuration"

if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ROOT/config.env.example" ]; then
        cp "$ROOT/config.env.example" "$ENV_FILE"
        ok "config.env created from config.env.example"
    else
        cat > "$ENV_FILE" <<EOF
ASSISTANT_NAME=MILO
JETSON_FRIEND_ROOT=$ROOT

LLM_MODEL_DIR=$MODELS/llm
VLM_MODEL_DIR=$MODELS/vlm

LLM_MODEL=$MODELS/llm/Qwen3-4B-Q4_K_M.gguf
VLM_MMPROJ=
LLM_ENABLE_VISION=auto

MODEL_AUTO_CAPABILITIES=1
MODEL_AUTO_RESTART=1
MODEL_AUTO_VISION=1
MODEL_REQUIRE_CHAT=1

LLAMA_BIN=$DEPS/llama.cpp/build/bin/llama-cli
LLAMA_SERVER_BIN=$DEPS/llama.cpp/build/bin/llama-server
LLAMA_SERVER_HOST=127.0.0.1
LLAMA_SERVER_PORT=8081
LLAMA_SERVER_URL=http://127.0.0.1:8081/v1/chat/completions
LLAMA_CTX=2048
LLAMA_GPU_LAYERS=99

LLM_TEMPERATURE=0.55
LLM_TOP_P=0.85
LLM_MAX_TOKENS=96
LLM_HISTORY_TURNS=2
LLM_TIMEOUT_SEC=60

WHISPER_BIN=$DEPS/whisper.cpp/build/bin/whisper-cli
WHISPER_MODEL=$MODELS/whisper/ggml-base.en.bin

PIPER_BIN=$VENV/bin/piper
PIPER_VOICE=$MODELS/tts/en_US-ryan-low.onnx

OBJECT_MODEL=$MODELS/vision/yolov8n.engine
FACE_MODEL=$MODELS/vision/face_detection_yunet.engine
COCO_LABELS=$MODELS/vision/coco.yaml
FACE_RECOGNITION_MODEL=$MODELS/vision/face_recognition_sface.onnx
EOF
        ok "Minimal config.env created"
    fi
else
    ok "Existing config.env preserved"
fi

chmod +x "$ROOT/start.sh" 2>/dev/null || true

log "Hardware discovery"

printf '\n--- Microphones ---\n'
arecord -l || warn "No ALSA microphone detected"

printf '\n--- Speakers ---\n'
aplay -l || warn "No ALSA playback device detected"

printf '\n--- Cameras ---\n'
find /dev -maxdepth 1 -name 'video*' -print 2>/dev/null || true

log "Code verification"

"$VENV/bin/python" -m py_compile \
    "$ROOT/src/main.py" \
    "$ROOT/src/ai.py" \
    "$ROOT/src/model_manager.py" \
    "$ROOT/src/model_downloader.py"

ok "Python source files compile"

log "INSTALLATION COMPLETE"

cat <<EOF

MILO is ready.

Normal start:
  ./start.sh

List available models:
  ./.venv/bin/python src/model_downloader.py list

Install the optional VLM:
  ./.venv/bin/python src/model_downloader.py download qwen2.5-vl-3b-q4

Installed models are discovered automatically by ModelManager.

EOF
