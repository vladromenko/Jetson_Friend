#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
chmod +x start_milo.sh stop_milo.sh diagnose.sh scripts/*.sh
set -a; source config.env; set +a
PYTHON="${APP_PYTHON:-python3}"; [ -x "$PYTHON" ] || PYTHON=python3
"$PYTHON" -m py_compile milo/*.py
"$PYTHON" - <<'PY'
required=('numpy','sounddevice','scipy','cv2','pygame')
missing=[]
for name in required:
    try: __import__(name)
    except Exception as exc: missing.append(f'{name}: {exc}')
if missing: raise SystemExit('[BLOCKED] missing Python dependencies: '+' | '.join(missing))
print('[READY] Python audio/vision/UI dependencies')
PY
python3 -m unittest discover -s tests -v
missing=0
for f in "$LLAMA_SERVER_BIN" "$LLM_MODEL" "$VLM_MMPROJ" "$WHISPER_BIN" "$WHISPER_MODEL" "$PIPER_BIN" "$PIPER_VOICE" "$FACE_MODEL"; do
  if [ ! -e "$f" ]; then echo "[MISSING] $f"; missing=1; fi
done
if [ "$missing" -ne 0 ]; then
  echo "[BLOCKED] required existing Jetson_Friend assets are missing"
  exit 1
fi
if [ -e "$FACE_ENGINE" ]; then echo "[READY] TensorRT YuNet available"; else echo "[INFO] TensorRT YuNet missing; ONNX/Haar fallback will be used"; fi
echo "[READY] MILO Clean RC13 installed"
echo "Start: cd $ROOT && ./start_milo.sh"
