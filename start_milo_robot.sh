#!/usr/bin/env bash
set -Ee -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/data"

ROS_DISTRO_DETECTED="${ROS_DISTRO:-}"
[ -z "$ROS_DISTRO_DETECTED" ] && [ -f /opt/ros/jazzy/setup.bash ] && ROS_DISTRO_DETECTED=jazzy
[ -z "$ROS_DISTRO_DETECTED" ] && [ -f /opt/ros/humble/setup.bash ] && ROS_DISTRO_DETECTED=humble
[ -n "$ROS_DISTRO_DETECTED" ] || { echo "ERROR: ROS2 not found"; exit 1; }

set +u
source "/opt/ros/$ROS_DISTRO_DETECTED/setup.bash"
[ -f "$ROOT/orbbec_ws/install/setup.bash" ] && source "$ROOT/orbbec_ws/install/setup.bash"
[ -f "$HOME/m3pro_hw_ws/install/setup.bash" ] && source "$HOME/m3pro_hw_ws/install/setup.bash"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"

cleanup() {
  [ -n "${ORBBEC_PID:-}" ] && kill "$ORBBEC_PID" 2>/dev/null || true
  [ -n "${AGENT_PID:-}" ] && kill "$AGENT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[MILO] Preparing STM32 micro-ROS connection"
if ! ros2 topic list 2>/dev/null | grep -qx '/arm_joint'; then
  AGENT="$(find "$HOME" -type f -path '*/micro_ros_agent' -perm -111 2>/dev/null | head -1 || true)"
  SERIAL="$(find /dev/serial/by-id -maxdepth 1 -type l -name '*CP2104*' 2>/dev/null | head -1 || true)"
  [ -z "$SERIAL" ] && SERIAL=/dev/ttyUSB0
  if [ -n "$AGENT" ] && [ -e "$SERIAL" ]; then
    "$AGENT" serial --dev "$SERIAL" -b 2000000 >"$ROOT/data/micro_ros_agent.log" 2>&1 &
    AGENT_PID=$!
    sleep 3
  fi
fi

if ! ros2 topic list 2>/dev/null | grep -qx '/arm_joint'; then
  echo "WARNING: /arm_joint is unavailable. Milo will start without arm motion."
fi

echo "[MILO] Preparing DaBai DCW2 arm RGB-D camera"
if ! ros2 topic list 2>/dev/null | grep -qx '/camera/color/image_raw'; then
  if ros2 pkg prefix orbbec_camera >/dev/null 2>&1; then
    ros2 launch orbbec_camera dabai_dcw2.launch.py align_mode:=HW align_target_stream:=COLOR >"$ROOT/data/orbbec_dabai.log" 2>&1 &
    ORBBEC_PID=$!
    sleep 7
  fi
fi

echo "[MILO] Logitech remains the existing global CAMERA_INDEX camera."
echo "[MILO] DaBai DCW2 is the RGB-D arm camera."
echo "[MILO] Checking required topics"
ARM_OK=0
COLOR_OK=0
DEPTH_OK=0
ros2 topic list 2>/dev/null | grep -qx '/arm_joint' && ARM_OK=1 || true
ros2 topic list 2>/dev/null | grep -qx '/camera/color/image_raw' && COLOR_OK=1 || true
ros2 topic list 2>/dev/null | grep -qx '/camera/depth/image_raw' && DEPTH_OK=1 || true
echo "[MILO] arm_joint=$ARM_OK color=$COLOR_OK depth=$DEPTH_OK"
echo "[MILO] Arm starts in SAFE mode. Say 'Milo, arm ready' before any arm movement."
echo "[MILO] Starting normal Milo with robotics integrated."
exec "$ROOT/start.sh" "$@"
