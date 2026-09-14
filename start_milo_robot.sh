#!/usr/bin/env bash
set -Ee -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/data"
# Load machine configuration before ROS and readiness checks.
set -a
[ ! -f "$ROOT/config.env" ] || source "$ROOT/config.env"
set +a
NO_MOTION=0
if [ "${1:-}" = --no-arm-motion ]; then NO_MOTION=1; shift; fi
export MILO_ARM_STARTUP_BLOCK_REASON=""
[ "$NO_MOTION" = 0 ] || export MILO_ARM_STARTUP_BLOCK_REASON="diagnostic run requested without arm motion"

# This descriptor stays open in child MILO, preventing another launcher/controller.
exec 9>"$ROOT/data/milo_robot.lock"
flock -n 9 || { echo 'ERROR: another MILO robot launcher owns this project'; exit 1; }
ROS_DISTRO_DETECTED="${ROS_DISTRO:-jazzy}"
[ -f "/opt/ros/$ROS_DISTRO_DETECTED/setup.bash" ] || { echo 'ERROR: ROS2 not found'; exit 1; }
set +u
source "/opt/ros/$ROS_DISTRO_DETECTED/setup.bash"
[ ! -f "$ROOT/orbbec_ws/install/setup.bash" ] || source "$ROOT/orbbec_ws/install/setup.bash"
[ ! -f "$HOME/m3pro_hw_ws/install/setup.bash" ] || source "$HOME/m3pro_hw_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
block_arm() {
  export MILO_ARM_STARTUP_BLOCK_REASON="${MILO_ARM_STARTUP_BLOCK_REASON:+$MILO_ARM_STARTUP_BLOCK_REASON; }$*"
  echo "[MILO] SAFE: $*"
}
cleanup() {
  trap - EXIT INT TERM
  for pid in "${MILO_PID:-}" "${ORBBEC_PID:-}" "${AGENT_PID:-}"; do
    [ -z "$pid" ] || kill -TERM -- "-$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

AGENT="${MILO_MICRO_ROS_AGENT:-$HOME/microros_ws/install/micro_ros_agent/lib/micro_ros_agent/micro_ros_agent}"
SERIAL="${MILO_ARM_SERIAL:-}"
if [ -z "$SERIAL" ]; then
  shopt -s nullglob
  ports=(/dev/serial/by-id/*CP2104*)
  shopt -u nullglob
  if [ "${#ports[@]}" = 1 ]; then SERIAL="${ports[0]}"; fi
fi
if [ -z "$SERIAL" ] || [ ! -e "$SERIAL" ]; then
  block_arm 'a unique CP2104 UART was not found; set MILO_ARM_SERIAL if needed'
else
  # Also serialize launchers from different checkouts against the same UART.
  SERIAL_REAL="$(readlink -f "$SERIAL")"
  exec 8>"/tmp/milo-uart-${SERIAL_REAL##*/}.lock"
  if ! flock -n 8; then
    block_arm 'another launcher owns the CP2104 UART'
  elif OWNER="$(python3 "$ROOT/tools/robot_serial_owner.py" "$SERIAL" "$ROS_DOMAIN_ID")"; then
    case "$OWNER" in
      reuse:*) echo "[MILO] Reusing micro_ros_agent PID ${OWNER#reuse:} on $SERIAL" ;;
      new)
        if [ -x "$AGENT" ]; then
          setsid "$AGENT" serial --dev "$SERIAL" -b 2000000 >"$ROOT/data/micro_ros_agent.log" 2>&1 &
          AGENT_PID=$!
          echo "[MILO] Started micro_ros_agent PID $AGENT_PID on $SERIAL"
        else block_arm "micro_ros_agent executable missing: $AGENT"; fi ;;
      *) block_arm 'unrecognized serial ownership result' ;;
    esac
  else block_arm 'conflicting or unverifiable UART ownership'; fi
fi
ARM_OK=0
if python3 "$ROOT/tools/robot_readiness.py" arm --timeout "${MILO_ARM_READY_TIMEOUT_SEC:-45}"; then
  ARM_OK=1
else block_arm 'arm subscribers/type support did not become ready before timeout'; fi

# Check actual streams, not just topic names in a possibly stale ROS CLI daemon.
CAMERA_OK=0
if python3 "$ROOT/tools/robot_readiness.py" camera --timeout 2; then
  CAMERA_OK=1
elif pgrep -f '[o]rbbec_camera|[c]omponent_container.*camera' >/dev/null; then
  echo '[MILO] Existing camera process found; waiting for its streams'
elif ros2 pkg prefix orbbec_camera >/dev/null 2>&1; then
  setsid ros2 launch orbbec_camera dabai_dcw2.launch.py align_mode:=HW align_target_stream:=COLOR >"$ROOT/data/orbbec_dabai.log" 2>&1 &
  ORBBEC_PID=$!
else block_arm 'DaBai driver is unavailable'; fi
if [ "$CAMERA_OK" = 0 ]; then
  if python3 "$ROOT/tools/robot_readiness.py" camera --timeout "${MILO_CAMERA_READY_TIMEOUT_SEC:-45}"; then
    CAMERA_OK=1
  else block_arm 'fresh DaBai color/depth did not become ready before timeout'; fi
fi
echo "[MILO] arm_joint=$ARM_OK color=$CAMERA_OK depth=$CAMERA_OK"
echo '[MILO] Logitech global camera and DaBai arm camera are separate inputs.'
if [ -z "$MILO_ARM_STARTUP_BLOCK_REASON" ]; then
  echo '[MILO] Face search will use the configured test pose and bounded J1 sector.'
else echo "[MILO] Autonomous arm motion DISABLED: $MILO_ARM_STARTUP_BLOCK_REASON"; fi
# Keep the launcher alive so EXIT cleanup runs for processes it owns.
setsid "$ROOT/start.sh" "$@" &
MILO_PID=$!
wait "$MILO_PID"
