#!/usr/bin/env bash
# Camera-only diagnostic; never starts an arm controller or micro-ROS agent.
set -Ee -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p data/manipulation_audit
set -a
[ ! -f config.env ] || source config.env
set +a
source /opt/ros/jazzy/setup.bash
source "$ROOT/orbbec_ws/install/setup.bash"
source "$HOME/m3pro_hw_ws/install/setup.bash"
source "$HOME/m3pro_hw_ws/install/arm_msgs/share/arm_msgs/local_setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export MILO_ARM_ENABLE=0 MILO_ARM_TRACKING_ENABLED=0
exec 9>data/manipulation_audit/dry_run.lock
flock -n 9 || { echo 'Another diagnostic is running'; exit 1; }
cleanup() {
  trap - EXIT INT TERM
  if [ -n "${CAMERA_PID:-}" ]; then kill -TERM -- "-$CAMERA_PID" 2>/dev/null || true; wait "$CAMERA_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! .venv/bin/python tools/robot_readiness.py camera --timeout 2; then
  if pgrep -f '[o]rbbec_camera|[c]omponent_container.*camera' >/dev/null; then
    echo 'Existing camera has no fresh streams; inspect its logs before retrying'; exit 1
  fi
  setsid ros2 launch orbbec_camera dabai_dcw2.launch.py align_mode:=HW align_target_stream:=COLOR > data/manipulation_audit/camera_diagnostic.log 2>&1 &
  CAMERA_PID=$!
  .venv/bin/python tools/robot_readiness.py camera --timeout 30
fi
.venv/bin/python tools/manipulation_dry_run.py --seconds "${1:-20}" --output data/manipulation_audit/live.json
