#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p deps models/vlm/gemma4 models/whisper models/tts models/vision
need_cmd(){ command -v "$1" >/dev/null 2>&1 || { echo "[MISSING CMD] $1"; exit 1; }; }
need_cmd git
need_cmd cmake
need_cmd make
need_cmd g++
need_cmd curl
LLAMA_COMMIT=335b21fcbda972777e4e9e69decad1f719cafeb3
WHISPER_COMMIT=307869af285d7f6f689ba100b3515e2d1b3feb05
checkout_pinned(){
  local url="$1" dir="$2" commit="$3"
  if [ ! -d "$dir/.git" ]; then git clone --filter=blob:none --no-checkout "$url" "$dir"; fi
  git -C "$dir" fetch --depth 1 origin "$commit"
  git -C "$dir" checkout --detach "$commit"
}
fetch(){
  local url="$1" out="$2"
  if [ -s "$out" ]; then echo "[READY] $out"; return; fi
  mkdir -p "$(dirname "$out")"
  echo "[FETCH] $out"
  curl -L --fail --retry 5 --continue-at - -o "$out" "$url"
}
if [ ! -x deps/llama.cpp/build/bin/llama-server ]; then
  checkout_pinned https://github.com/ggml-org/llama.cpp deps/llama.cpp "$LLAMA_COMMIT"
  cmake -S deps/llama.cpp -B deps/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build deps/llama.cpp/build --target llama-server -j"$(nproc)"
fi
if [ ! -x deps/whisper.cpp/build/bin/whisper-cli ]; then
  checkout_pinned https://github.com/ggml-org/whisper.cpp deps/whisper.cpp "$WHISPER_COMMIT"
  cmake -S deps/whisper.cpp -B deps/whisper.cpp/build -DGGML_CUDA=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build deps/whisper.cpp/build --target whisper-cli -j"$(nproc)"
fi
fetch "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf" "models/vlm/gemma4/gemma-4-E2B-it-Q4_0.gguf"
fetch "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B-it-mmproj.gguf" "models/vlm/gemma4/mmproj-gemma-4-E2B-it-Q8_0.gguf"
fetch "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" "models/whisper/ggml-base.en.bin"
fetch "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx" "models/tts/en_US-ryan-low.onnx"
fetch "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/low/en_US-ryan-low.onnx.json" "models/tts/en_US-ryan-low.onnx.json"
fetch "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" "models/vision/face_detection_yunet.onnx"
chmod +x deps/llama.cpp/build/bin/llama-server deps/whisper.cpp/build/bin/whisper-cli 2>/dev/null || true
echo "[READY] assets bootstrap complete"
