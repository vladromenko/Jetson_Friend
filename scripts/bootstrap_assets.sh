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
  local url="$1" out="$2" expected="$3" partial="${2}.partial"
  if [ -s "$out" ]; then
    echo "$expected  $out" | sha256sum --check --status || {
      echo "[BLOCKED] checksum mismatch: $out"
      exit 1
    }
    echo "[READY] verified $out"
    return
  fi
  mkdir -p "$(dirname "$out")"
  echo "[FETCH] $out"
  curl -L --fail --retry 5 --continue-at - -o "$partial" "$url"
  echo "$expected  $partial" | sha256sum --check --status || {
    echo "[BLOCKED] downloaded checksum mismatch: $out"
    exit 1
  }
  mv "$partial" "$out"
}
if [ ! -x deps/llama.cpp/build/bin/llama-server ]; then
  checkout_pinned https://github.com/ggml-org/llama.cpp deps/llama.cpp "$LLAMA_COMMIT"
  cmake -S deps/llama.cpp -B deps/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build deps/llama.cpp/build --target llama-server -j"${MILO_BUILD_JOBS:-2}"
fi
if [ ! -x deps/whisper.cpp/build/bin/whisper-cli ]; then
  checkout_pinned https://github.com/ggml-org/whisper.cpp deps/whisper.cpp "$WHISPER_COMMIT"
  cmake -S deps/whisper.cpp -B deps/whisper.cpp/build -DGGML_CUDA=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build deps/whisper.cpp/build --target whisper-cli -j"${MILO_BUILD_JOBS:-2}"
fi
fetch "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf" \
  "models/vlm/gemma4/gemma-4-E2B-it-Q4_0.gguf" \
  "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634"
fetch "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B-it-mmproj.gguf" \
  "models/vlm/gemma4/mmproj-gemma-4-E2B-it-Q8_0.gguf" \
  "021059cce659fe7f9170d5599761d7bbaf644b798dab9503aca30dc43e6beb14"
fetch "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" \
  "models/whisper/ggml-base.en.bin" \
  "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"
fetch "https://huggingface.co/rhasspy/piper-voices/resolve/c10ece1aade47bb51c153c893d14e5bf8e5b7117/en/en_US/ryan/low/en_US-ryan-low.onnx" \
  "models/tts/en_US-ryan-low.onnx" \
  "8d21a085cc4c0010f1f3e91d5008c8691277ccfa744eb0d747becd33a3444baf"
fetch "https://huggingface.co/rhasspy/piper-voices/resolve/c10ece1aade47bb51c153c893d14e5bf8e5b7117/en/en_US/ryan/low/en_US-ryan-low.onnx.json" \
  "models/tts/en_US-ryan-low.onnx.json" \
  "b27147e56b0525962609f82f58171f4618cbf17c6fb043d7d724ff28cc4aed60"
fetch "https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  "models/vision/face_detection_yunet.onnx" \
  "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
chmod +x deps/llama.cpp/build/bin/llama-server deps/whisper.cpp/build/bin/whisper-cli 2>/dev/null || true
echo "[READY] assets bootstrap complete"
