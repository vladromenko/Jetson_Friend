#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs data
set -a; source "$ROOT/config.env"; set +a
export ROS_DOMAIN_ID
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DISPLAY="${DISPLAY:-${DISPLAY_NAME:-:0}}"
PYTHON="${APP_PYTHON:-python3}"
[ -x "$PYTHON" ] || PYTHON=python3

if [ -x "$ROOT/scripts/set_display.sh" ]; then
  "$ROOT/scripts/set_display.sh" "${MILO_DISPLAY_ROTATION:-left}" >/dev/null 2>&1 || true
fi

exec 9>"$ROOT/data/milo.lock"
flock -n 9 || { echo '[FATAL] MILO is already running'; exit 1; }
if pgrep -f '[p]ython.*-m milo.main' >/dev/null; then
  echo '[FATAL] another MILO version is running; stop it before starting this one'
  exit 1
fi

OWNED=()
spawn_logged(){ local log="$1"; shift; setsid "$@" >"$log" 2>&1 & OWNED+=("$!"); }
cleanup(){
  local code=$?
  trap - EXIT INT TERM
  for p in "${OWNED[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
  sleep 0.5
  for p in "${OWNED[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +u
source /opt/ros/jazzy/setup.bash
declare -A SOURCED_WORKSPACES=()
for workspace in "$INTERFACES_WS" "$MICROROS_WS" "$ORBBEC_WS"; do
  [ -n "${SOURCED_WORKSPACES[$workspace]:-}" ] && continue
  [ ! -f "$workspace/install/setup.bash" ] || source "$workspace/install/setup.bash"
  SOURCED_WORKSPACES[$workspace]=1
done
set -u

if [ ! -e "$ARM_SERIAL" ]; then
  echo "[DEGRADED] CP2104 not found; communication/vision still run, arm motion disabled"
elif ! pgrep -f '[m]icro_ros_agent.*serial.*2000000' >/dev/null; then
  spawn_logged "$ROOT/logs/micro_ros_agent.log" ros2 run micro_ros_agent micro_ros_agent serial --dev "$ARM_SERIAL" -b "$ARM_BAUD"
  echo "[START] micro-ROS agent"
  sleep 2
else
  echo "[READY] micro-ROS agent"
fi

camera_frame_ready(){ timeout 4 ros2 topic echo "$CAMERA_COLOR_TOPIC" --once >/dev/null 2>&1; }
start_camera(){
  spawn_logged "$ROOT/logs/dabai.log" ros2 launch "$ORBBEC_WS/src/OrbbecSDK_ROS2/orbbec_camera/launch/dabai_dcw2.launch.py"
  echo "[START] DaBai camera"
}

if pgrep -f '[o]rbbec_camera|[c]omponent_container.*camera' >/dev/null; then
  if camera_frame_ready; then
    echo "[READY] DaBai already publishing"
  else
    echo "[RECOVER] stale DaBai process; restarting camera"
    pkill -INT -f '[o]rbbec_camera|[c]omponent_container.*camera' 2>/dev/null || true
    sleep 2
    start_camera
  fi
else
  start_camera
fi

echo "[WAIT] DaBai color/depth"
color_ready=0
depth_ready=0
for _ in $(seq 1 35); do
  if [ "$color_ready" -eq 0 ] && timeout 2 ros2 topic echo "$CAMERA_COLOR_TOPIC" --once >/dev/null 2>&1; then color_ready=1; fi
  if [ "$depth_ready" -eq 0 ] && timeout 2 ros2 topic echo "$CAMERA_DEPTH_TOPIC" --once >/dev/null 2>&1; then depth_ready=1; fi
  if [ "$color_ready" -eq 1 ] && [ "$depth_ready" -eq 1 ]; then break; fi
  sleep 1
done
[ "$color_ready" -eq 1 ] && echo "[READY] DaBai color" || echo "[DEGRADED] DaBai color unavailable"
[ "$depth_ready" -eq 1 ] && echo "[READY] DaBai depth" || echo "[DEGRADED] DaBai depth unavailable"

start_model_server(){
  spawn_logged "$ROOT/logs/llm.log" "$ROOT/scripts/serve_model.sh"
  echo "[START] local Gemma + mmproj"
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://${LLM_HOST}:${LLM_PORT}/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    echo "[FATAL] local Gemma server failed; see logs/llm.log"
    exit 1
  fi
  echo "[READY] local Gemma server"
}

if curl -fsS "http://${LLM_HOST}:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "[READY] existing local model server found"
  if ! "$PYTHON" -m milo.preflight --llm --vlm; then
    echo "[RECOVER] existing model server failed text/vision preflight; restarting on port ${LLM_PORT}"
    if command -v fuser >/dev/null 2>&1; then fuser -k "${LLM_PORT}/tcp" >/dev/null 2>&1 || true; else pkill -TERM -f '[l]lama-server' 2>/dev/null || true; fi
    sleep 2
    start_model_server
  fi
else
  start_model_server
fi

echo "[CHECK] model text + vision response"
"$PYTHON" -m milo.preflight --llm --vlm

echo "[CHECK] audio devices"
"$PYTHON" -m milo.preflight --audio

echo "[CHECK] application-level DaBai + face detector"
if ! "$PYTHON" -m milo.preflight --vision; then
  echo "[DEGRADED] application camera preflight failed; voice can still run, tracking will stay disabled"
fi

echo "[CHECK] arm feedback (J2 raised at 115; J4/J6 locked in MILO)"
timeout 5 ros2 topic echo "$ARM_FEEDBACK_TOPIC" --once || true

echo "[START] MILO Clean RC13"
"$PYTHON" -m milo.main
