#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-install}"

if [ "$MODE" != install ] && [ "$MODE" != --check ]; then
  echo "usage: ./install.sh [--check]"
  exit 2
fi

if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "[BLOCKED] ROS 2 Jazzy is required at /opt/ros/jazzy"
  exit 1
fi

CONFIG_FILE="$ROOT/config.env"
if [ ! -f "$CONFIG_FILE" ] && [ "$MODE" = install ]; then
  cp config.env.example config.env
  echo "[READY] created config.env from the checked-in example"
elif [ ! -f "$CONFIG_FILE" ]; then
  CONFIG_FILE="$ROOT/config.env.example"
  echo "[INFO] config.env is absent; checking the canonical example"
fi

if [ "$MODE" = install ]; then
  chmod +x start_milo.sh stop_milo.sh diagnose.sh scripts/*.sh
  ./scripts/bootstrap_system.sh
  if [ ! -x .venv/bin/python ]; then
    python3 -m venv --system-site-packages .venv
  fi
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
  ./scripts/bootstrap_ros.sh
  ./scripts/bootstrap_assets.sh
  install -Dm644 milo.service "$HOME/.config/systemd/user/milo.service"
  systemctl --user daemon-reload
  systemctl --user disable milo.service >/dev/null 2>&1 || true
fi

set -a
source "$CONFIG_FILE"
set +a

set +u
source /opt/ros/jazzy/setup.bash
source "$ROOT/ros_ws/install/setup.bash"
set -u

PYTHON="${APP_PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || { echo "[BLOCKED] Python environment is missing; run ./install.sh"; exit 1; }

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" - <<'PY'
import ast
from pathlib import Path
for path in Path('milo').glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
print('[READY] Python syntax')
PY
"$PYTHON" - <<'PY'
required = ('numpy', 'sounddevice', 'scipy', 'cv2', 'pygame', 'rclpy')
missing = []
for name in required:
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f'{name}: {exc}')
if missing:
    raise SystemExit('[BLOCKED] missing Python dependencies: ' + ' | '.join(missing))
print('[READY] Python audio/vision/UI/ROS dependencies')
PY

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -q -p no:cacheprovider

missing=0
for file in \
  "$LLAMA_SERVER_BIN" "$LLM_MODEL" "$VLM_MMPROJ" \
  "$WHISPER_BIN" "$WHISPER_MODEL" "$PIPER_BIN" "$PIPER_VOICE" \
  "$FACE_MODEL" "$ROOT/ros_ws/install/setup.bash"; do
  if [ ! -e "$file" ]; then echo "[MISSING] $file"; missing=1; fi
done
if [ "$missing" -ne 0 ]; then
  echo "[BLOCKED] runtime dependencies are incomplete; run ./install.sh"
  exit 1
fi

if [ -e "$FACE_ENGINE" ]; then
  echo "[READY] TensorRT YuNet available"
else
  echo "[INFO] TensorRT YuNet missing; ONNX/Haar fallback will be used"
fi

echo "[READY] MILO RC13 is installed and verified"
echo "[STOPPED] installation does not start MILO"
echo "Start manually: cd $ROOT && ./start_milo.sh"
