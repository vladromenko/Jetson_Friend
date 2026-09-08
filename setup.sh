#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$ROOT/models"
ENV_FILE="$ROOT/config.env"

LLM_REPO="${LLM_REPO:-Qwen/Qwen3-4B-GGUF}"
LLM_FILE="${LLM_FILE:-Qwen3-4B-Q4_K_M.gguf}"
WHISPER_MODEL="${WHISPER_MODEL:-base.en}"
PIPER_VOICE="${PIPER_VOICE:-en_US-ryan-low}"
YOLO_URL="${YOLO_URL:-https://raw.githubusercontent.com/yoobright/yolo-onnx/master/yolov8n.onnx}"
COCO_URL="${COCO_URL:-https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml}"
FACE_URL="${FACE_URL:-https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx}"

log(){ printf '\n== %s ==\n' "$*"; }
ok(){ printf 'OK: %s\n' "$*"; }
warn(){ printf 'WARN: %s\n' "$*" >&2; }
need(){ command -v "$1" >/dev/null 2>&1; }

download(){
  local url="$1" out="$2"
  [[ -s "$out" ]] && { ok "$(basename "$out") exists"; return; }
  mkdir -p "$(dirname "$out")"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$out" "$url"
}

detect_platform(){
  log "Platform"
  ARCH="$(uname -m)"
  UBUNTU="$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
  L4T="$(dpkg-query -W -f='${Version}' nvidia-l4t-core 2>/dev/null || true)"
  JETPACK="$(apt-cache show nvidia-jetpack 2>/dev/null | awk '/^Version:/{print $2; exit}' || true)"
  MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
  [[ "$ARCH" == "aarch64" ]] || warn "Not ARM64/aarch64. This project is intended for Jetson."
  [[ "$MODEL" == *"Jetson"* ]] || warn "Jetson model not detected on this host."
  printf 'arch=%s\nubuntu=%s\njetson=%s\nl4t=%s\njetpack=%s\n' "$ARCH" "$UBUNTU" "${MODEL:-unknown}" "${L4T:-unknown}" "${JETPACK:-unknown}"
}

detect_nvidia(){
  log "NVIDIA stack"
  need nvcc && nvcc --version | sed -n 's/^.*release /CUDA /p' | head -1 || warn "CUDA compiler not found"
  ldconfig -p 2>/dev/null | grep -q libcudnn && ok "cuDNN found" || warn "cuDNN not found"
  need trtexec && trtexec --version 2>/dev/null | head -1 || warn "TensorRT trtexec not found"
}

install_host_packages(){
  log "Host packages"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential cmake pkg-config \
    alsa-utils pulseaudio-utils v4l-utils x11-xserver-utils \
    docker-compose-plugin python3
  sudo apt-get install -y --no-install-recommends nvidia-container-toolkit >/dev/null 2>&1 \
    || sudo apt-get install -y --no-install-recommends nvidia-container-runtime >/dev/null 2>&1 \
    || warn "NVIDIA container runtime package not installed from current apt sources"
  ok "required host packages present"
}

detect_docker(){
  log "Docker"
  sudo systemctl enable --now docker >/dev/null 2>&1 || true
  sudo usermod -aG docker "$USER" || true
  docker --version
  if ! docker info >/dev/null 2>&1; then
    warn "Docker needs a new login or sudo permissions. Trying sudo docker for setup."
    DOCKER="sudo docker"
  else
    DOCKER="docker"
  fi
  if ! dpkg -l | grep -q nvidia-container-toolkit; then
    warn "nvidia-container-toolkit is missing. Install it from NVIDIA Jetson repos for GPU containers."
  fi
}

write_config(){
  log "Config"
  if [[ ! -f "$ENV_FILE" ]]; then
    cp "$ROOT/config.env.example" "$ENV_FILE"
    ok "created config.env"
  else
    ok "config.env exists"
  fi
  mkdir -p "$MODELS"/{llm,whisper,tts,vision}
}

download_models(){
  log "Models"
  download "https://huggingface.co/${LLM_REPO}/resolve/main/${LLM_FILE}?download=true" "$MODELS/llm/$LLM_FILE"
  download "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${WHISPER_MODEL}.bin?download=true" "$MODELS/whisper/ggml-${WHISPER_MODEL}.bin"
  download "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/${PIPER_VOICE}.onnx?download=true" "$MODELS/tts/${PIPER_VOICE}.onnx"
  download "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/${PIPER_VOICE}.onnx.json?download=true" "$MODELS/tts/${PIPER_VOICE}.onnx.json"
  download "$YOLO_URL" "$MODELS/vision/yolov8n.onnx"
  download "$FACE_URL" "$MODELS/vision/face_detection_yunet.onnx"
  download "$COCO_URL" "$MODELS/vision/coco.yaml"
}

build_container(){
  log "Container"
  local base="nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04"
  $DOCKER build --build-arg BASE_IMAGE="$base" -t jetson-friend:local "$ROOT"
}

build_trt_engines(){
  log "TensorRT engines"
  if need trtexec; then
    [[ -s "$MODELS/vision/yolov8n.engine" ]] || trtexec --onnx="$MODELS/vision/yolov8n.onnx" --saveEngine="$MODELS/vision/yolov8n.engine" --fp16 --workspace=1024 >/tmp/jetson_friend_trt.log 2>&1 || warn "YOLO TensorRT build failed; runtime will use ONNX fallback"
    [[ -s "$MODELS/vision/face_detection_yunet.engine" ]] || trtexec --onnx="$MODELS/vision/face_detection_yunet.onnx" --saveEngine="$MODELS/vision/face_detection_yunet.engine" --fp16 --workspace=512 >/tmp/jetson_friend_face_trt.log 2>&1 || warn "Face TensorRT build failed; runtime will use ONNX fallback"
  else
    warn "trtexec unavailable on host; skipping engine build"
  fi
}

detect_devices(){
  log "Devices"
  ls /dev/video* 2>/dev/null || warn "No camera device found"
  arecord -l 2>/dev/null || warn "No microphone found"
  aplay -l 2>/dev/null || warn "No speaker found"
  xrandr --current 2>/dev/null | awk '/ connected/{print "display="$1" "$3}' || warn "No X display detected"
}

smoke(){
  log "Smoke tests"
  $DOCKER run --rm --runtime nvidia jetson-friend:local bash -lc 'test -e /dev/nvhost-ctrl-gpu || test -e /dev/nvhost-gpu || nvidia-smi >/dev/null 2>&1 || exit 1' && ok "GPU visible in container" || warn "GPU not verified in container"
  $DOCKER run --rm -v "$MODELS:/app/models" jetson-friend:local bash -lc 'test -x /app/deps/llama.cpp/build/bin/llama-cli && test -x /app/deps/whisper.cpp/build/bin/whisper-cli && command -v piper'
  ok "runtime binaries present"
}

detect_platform
detect_nvidia
install_host_packages
detect_docker
write_config
download_models
build_container
build_trt_engines
detect_devices
smoke

cat <<EOF

DONE:
setup complete enough to run ./start.sh

NOTE:
If Docker group membership changed, log out/in or run: newgrp docker
EOF
