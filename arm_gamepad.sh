#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$ROOT"
mkdir -p logs data
set -a; source "$ROOT/config.env"; set +a
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SDL_VIDEODRIVER=dummy
PYTHON="${APP_PYTHON:-python3}"
[ -x "$PYTHON" ] || PYTHON=python3

exec 9>"$ROOT/data/milo.lock"
flock -n 9 || { echo "[BLOCKED] MILO is running. Stop it before manual control."; exit 1; }
if pgrep -f '[p]ython.*-m milo.main' >/dev/null; then
  echo "[BLOCKED] MILO is running. Stop it before manual control."
  exit 1
fi

OWNED=()
cleanup(){
  local code=$?
  trap - EXIT INT TERM
  for pid in "${OWNED[@]}"; do kill -INT -- "-$pid" 2>/dev/null || true; done
  sleep 0.5
  for pid in "${OWNED[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
spawn_logged(){ local log="$1"; shift; setsid "$@" >"$log" 2>&1 & OWNED+=("$!"); }

set +u
source /opt/ros/jazzy/setup.bash
source "$INTERFACES_WS/install/setup.bash"
source "$MICROROS_WS/install/setup.bash"
source "$ORBBEC_WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID

if [ ! -e "$ARM_SERIAL" ]; then
  echo "[BLOCKED] STM32 serial port unavailable: $ARM_SERIAL"
  exit 1
fi
if ! pgrep -f '[m]icro_ros_agent.*serial.*2000000' >/dev/null; then
  spawn_logged "$ROOT/logs/gamepad_micro_ros_agent.log" \
    ros2 run micro_ros_agent micro_ros_agent serial --dev "$ARM_SERIAL" -b "$ARM_BAUD"
  echo "[START] micro-ROS agent"
fi

if ! timeout 4 ros2 topic echo "$CAMERA_COLOR_TOPIC" --once >/dev/null 2>&1; then
  spawn_logged "$ROOT/logs/gamepad_dabai.log" \
    ros2 launch "$ORBBEC_WS/src/OrbbecSDK_ROS2/orbbec_camera/launch/dabai_dcw2.launch.py"
  echo "[START] DaBai camera"
fi

echo "[START] six-servo gamepad. START enables motion; SELECT stops motion."
echo "[START] Ctrl+C ends the program. Demonstrations: $ROOT/data/candy_demos"
"$PYTHON" "$ROOT/arm_gamepad.py"
