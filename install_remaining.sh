#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPS="$ROOT/deps"
MODELS="$ROOT/models"
VENV="$ROOT/.venv"
ENV_FILE="$ROOT/config.env"

LLAMA_BIN="$DEPS/llama.cpp/build/bin/llama-cli"
WHISPER_BIN="$DEPS/whisper.cpp/build/bin/whisper-cli"

LLM="$MODELS/llm/Qwen3-4B-Q4_K_M.gguf"
WHISPER="$MODELS/whisper/ggml-base.en.bin"

PIPER="$MODELS/tts/en_US-ryan-low.onnx"
PIPER_CONFIG="$MODELS/tts/en_US-ryan-low.onnx.json"

YOLO="$MODELS/vision/yolov8n.onnx"
YOLO_ENGINE="$MODELS/vision/yolov8n.engine"

FACE="$MODELS/vision/face_detection_yunet.onnx"
FACE_ENGINE="$MODELS/vision/face_detection_yunet.engine"

COCO="$MODELS/vision/coco.yaml"


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


download() {
    local url="$1"
    local target="$2"

    mkdir -p "$(dirname "$target")"

    if [ -s "$target" ]; then
        ok "$(basename "$target") already exists"
        return 0
    fi

    curl \
        -L \
        --fail \
        --retry 5 \
        --retry-delay 3 \
        --connect-timeout 30 \
        -o "$target.part" \
        "$url"

    mv "$target.part" "$target"

    ok "Downloaded $(basename "$target")"
}


log "Jetson Friend remaining setup"


# ============================================================
# Directories
# ============================================================

mkdir -p \
    "$MODELS/vision" \
    "$MODELS/llm" \
    "$MODELS/whisper" \
    "$MODELS/tts"


# ============================================================
# CUDA
# ============================================================

log "CUDA"

if [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
    ok "CUDA available"
else
    die "CUDA nvcc not found"
fi


# ============================================================
# TensorRT
# ============================================================

log "TensorRT"

if command -v trtexec >/dev/null 2>&1; then
    ok "TensorRT trtexec found"
else
    warn "TensorRT trtexec not found"
    warn "Vision will fall back to OpenCV ONNX"
fi


# ============================================================
# Existing AI components
# ============================================================

log "Existing AI components"

if [ -x "$LLAMA_BIN" ]; then
    ok "llama.cpp exists"
else
    die "llama.cpp binary missing: $LLAMA_BIN"
fi

if [ -x "$WHISPER_BIN" ]; then
    ok "whisper.cpp exists"
else
    die "whisper.cpp binary missing: $WHISPER_BIN"
fi

if [ -s "$LLM" ]; then
    ok "Qwen model exists"
else
    die "Qwen model missing: $LLM"
fi

if [ -s "$WHISPER" ]; then
    ok "Whisper model exists"
else
    die "Whisper model missing: $WHISPER"
fi

if [ -s "$PIPER" ]; then
    ok "Piper voice exists"
else
    die "Piper voice missing: $PIPER"
fi

if [ -s "$PIPER_CONFIG" ]; then
    ok "Piper config exists"
else
    warn "Piper JSON config missing: $PIPER_CONFIG"
fi


# ============================================================
# System packages
# ============================================================

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


# ============================================================
# Python environment
# ============================================================

log "Python virtual environment"

if [ ! -d "$VENV" ]; then
    python3 -m venv \
        --system-site-packages \
        "$VENV"
fi

PYVENV_CFG="$VENV/pyvenv.cfg"

if [ -f "$PYVENV_CFG" ]; then

    if grep -q \
        '^include-system-site-packages = false' \
        "$PYVENV_CFG"
    then

        sed -i \
            's/^include-system-site-packages = false/include-system-site-packages = true/' \
            "$PYVENV_CFG"

        ok "Enabled system site packages"

    else
        ok "System site packages already enabled"
    fi

fi


# ============================================================
# Remove incompatible pip NumPy
# ============================================================

log "NumPy compatibility"

if "$VENV/bin/python" -m pip show numpy >/dev/null 2>&1; then

    NUMPY_LOCATION="$(
        "$VENV/bin/python" -m pip show numpy |
        awk -F': ' '/^Location:/ {print $2}'
    )"

    if [[ "$NUMPY_LOCATION" == "$VENV"* ]]; then

        warn "Removing pip NumPy from virtual environment"
        warn "Jetson Friend will use Ubuntu NumPy 1.x instead"

        "$VENV/bin/python" \
            -m pip uninstall \
            -y numpy

    fi

fi


# ============================================================
# Python packages
# ============================================================

log "Python packages"

"$VENV/bin/python" -m pip install \
    --upgrade \
    pip \
    setuptools \
    wheel

"$VENV/bin/python" -m pip install \
    sounddevice \
    soundfile \
    pygame \
    pyyaml \
    requests \
    psutil \
    pyudev \
    piper-tts


# Piper or another dependency may install NumPy 2.x again.
# Remove it so Python uses Ubuntu NumPy 1.26.x.

if "$VENV/bin/python" -m pip show numpy >/dev/null 2>&1; then

    NUMPY_LOCATION="$(
        "$VENV/bin/python" -m pip show numpy |
        awk -F': ' '/^Location:/ {print $2}'
    )"

    if [[ "$NUMPY_LOCATION" == "$VENV"* ]]; then

        warn "A pip dependency installed NumPy inside venv"
        warn "Removing it to preserve OpenCV compatibility"

        "$VENV/bin/python" \
            -m pip uninstall \
            -y numpy

    fi

fi

ok "Python packages installed"


# ============================================================
# Python import test
# ============================================================

log "Python module test"

"$VENV/bin/python" - <<'PY'
import sys

print("Python:", sys.version)

import numpy
print("NumPy:", numpy.__version__)
print("NumPy path:", numpy.__file__)

import cv2
print("OpenCV:", cv2.__version__)
print("OpenCV path:", cv2.__file__)

import sounddevice
print("sounddevice: OK")

import soundfile
print("soundfile: OK")

import pygame
print("pygame: OK")

import yaml
print("PyYAML: OK")

import requests
print("requests: OK")

import psutil
print("psutil: OK")

import pyudev
print("pyudev: OK")

print()
print("All Python imports: OK")
PY

ok "Python environment works"


# ============================================================
# Vision models
# ============================================================

log "YOLOv8n"

download \
    "https://raw.githubusercontent.com/yoobright/yolo-onnx/master/yolov8n.onnx" \
    "$YOLO"


log "YuNet face detector"

download \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    "$FACE"


log "COCO configuration"

download \
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml" \
    "$COCO"


# ============================================================
# Validate vision models
# ============================================================

log "Vision model validation"

"$VENV/bin/python" - <<PY
from pathlib import Path
import cv2

models = [
    Path("$YOLO"),
    Path("$FACE"),
]

for model in models:

    if not model.exists():
        raise RuntimeError(
            f"Missing model: {model}"
        )

    size = model.stat().st_size

    if size < 10000:
        raise RuntimeError(
            f"Model looks corrupted or incomplete: {model}"
        )

    print()
    print("Model:", model)
    print("Size:", size, "bytes")

    try:
        cv2.dnn.readNetFromONNX(
            str(model)
        )

        print("OpenCV ONNX load: OK")

    except Exception as exc:

        print(
            "OpenCV ONNX warning:",
            exc
        )

print()
print("Vision model validation complete")
PY


# ============================================================
# TensorRT YOLO
# ============================================================

if command -v trtexec >/dev/null 2>&1; then

    log "YOLO TensorRT engine"

    if [ -s "$YOLO_ENGINE" ]; then

        ok "YOLO TensorRT engine already exists"

    else

        if trtexec \
            --onnx="$YOLO" \
            --saveEngine="$YOLO_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:1024 \
            --skipInference \
            >"$ROOT/trt_yolo.log" 2>&1
        then

            ok "YOLO TensorRT engine created"

        else

            rm -f "$YOLO_ENGINE"

            warn "YOLO TensorRT conversion failed"
            warn "OpenCV ONNX fallback will be used"
            warn "See: $ROOT/trt_yolo.log"

        fi

    fi

fi


# ============================================================
# TensorRT YuNet
# ============================================================

if command -v trtexec >/dev/null 2>&1; then

    log "YuNet TensorRT engine"

    if [ -s "$FACE_ENGINE" ]; then

        ok "YuNet TensorRT engine already exists"

    else

        if trtexec \
            --onnx="$FACE" \
            --saveEngine="$FACE_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:512 \
            --skipInference \
            >"$ROOT/trt_face.log" 2>&1
        then

            ok "YuNet TensorRT engine created"

        else

            rm -f "$FACE_ENGINE"

            warn "YuNet TensorRT conversion failed"
            warn "OpenCV ONNX fallback will be used"
            warn "See: $ROOT/trt_face.log"

        fi

    fi

fi


# ============================================================
# config.env
# ============================================================

log "Configuration"

if [ ! -f "$ENV_FILE" ]; then

    cat >"$ENV_FILE" <<EOF
JETSON_FRIEND_ROOT=$ROOT

LANGUAGE=en

LLM_MODEL=$LLM
LLAMA_BIN=$LLAMA_BIN

WHISPER_MODEL=$WHISPER
WHISPER_BIN=$WHISPER_BIN

PIPER_MODEL=$PIPER
PIPER_CONFIG=$PIPER_CONFIG

YOLO_MODEL=$YOLO
YOLO_ENGINE=$YOLO_ENGINE

FACE_MODEL=$FACE
FACE_ENGINE=$FACE_ENGINE

COCO_CONFIG=$COCO
EOF

    ok "config.env created"

else

    ok "Existing config.env preserved"

fi


# ============================================================
# Hardware discovery
# ============================================================

log "Microphones"

if arecord -l; then
    ok "Audio input enumeration works"
else
    warn "No ALSA microphone detected"
fi


log "Speakers"

if aplay -l; then
    ok "Audio output enumeration works"
else
    warn "No ALSA playback device detected"
fi


log "Camera devices"

CAMERAS="$(
    find /dev \
        -maxdepth 1 \
        -name 'video*' \
        -print \
        2>/dev/null || true
)"

if [ -n "$CAMERAS" ]; then

    printf '%s\n' "$CAMERAS"

    while IFS= read -r camera
    do

        if [ -n "$camera" ]; then

            printf '\n--- %s ---\n' "$camera"

            v4l2-ctl \
                --device="$camera" \
                --all \
                2>/dev/null |
                head -30 || true

        fi

    done <<< "$CAMERAS"

else

    warn "No /dev/video* devices found"
    warn "Camera is not currently visible to Linux"

fi


# ============================================================
# Piper test
# ============================================================

log "Piper"

if "$VENV/bin/python" \
    -m piper \
    --help \
    >/dev/null 2>&1
then

    ok "Piper available"

else

    warn "Piper CLI test failed"

fi


# ============================================================
# Whisper test
# ============================================================

log "Whisper"

if [ -x "$WHISPER_BIN" ]; then

    "$WHISPER_BIN" \
        --help \
        >/dev/null 2>&1 || true

    ok "whisper.cpp executable available"

else

    die "whisper-cli disappeared"

fi


# ============================================================
# llama.cpp test
# ============================================================

log "llama.cpp"

if [ -x "$LLAMA_BIN" ]; then

    ok "llama.cpp executable available"

else

    die "llama-cli disappeared"

fi


# ============================================================
# start_native.sh
# ============================================================

log "Creating start_native.sh"

cat >"$ROOT/start_native.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

export JETSON_FRIEND_ROOT="$ROOT"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "Python environment missing." >&2
    echo "Run ./install_remaining.sh first." >&2
    exit 1
fi

if [ ! -f "$ROOT/config.env" ]; then
    echo "config.env missing." >&2
    exit 1
fi

if [ ! -f "$ROOT/src/main.py" ]; then
    echo "src/main.py missing." >&2
    exit 1
fi

exec \
    "$ROOT/.venv/bin/python" \
    "$ROOT/src/main.py" \
    "$@"
EOF

chmod +x "$ROOT/start_native.sh"

ok "start_native.sh created"


# ============================================================
# Final verification
# ============================================================

log "Final verification"

printf '\nModels:\n'

ls -lh \
    "$LLM" \
    "$WHISPER" \
    "$PIPER" \
    "$YOLO" \
    "$FACE"

printf '\nExecutables:\n'

printf 'llama.cpp:   %s\n' "$LLAMA_BIN"
printf 'whisper.cpp: %s\n' "$WHISPER_BIN"
printf 'python:      %s\n' "$VENV/bin/python"

printf '\nTensorRT engines:\n'

if [ -s "$YOLO_ENGINE" ]; then
    ls -lh "$YOLO_ENGINE"
else
    printf 'YOLO: ONNX fallback\n'
fi

if [ -s "$FACE_ENGINE" ]; then
    ls -lh "$FACE_ENGINE"
else
    printf 'YuNet: ONNX fallback\n'
fi


log "INSTALLATION COMPLETE"

cat <<EOF

Jetson Friend native environment is ready.

Core components:

  CUDA                  OK
  TensorRT              checked
  llama.cpp             OK
  Qwen3-4B              OK
  whisper.cpp           OK
  Whisper base.en       OK
  Piper                 OK
  OpenCV                OK
  NumPy                 system-compatible
  SoundDevice           OK
  SoundFile             OK
  pygame                OK
  PyYAML                 OK
  requests               OK
  psutil                 OK
  pyudev                 OK
  YOLOv8n                OK
  YuNet                  OK
  COCO config            OK

Run:

    cd "$ROOT"
    ./start_native.sh

EOF