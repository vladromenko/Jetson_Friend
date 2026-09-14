#!/usr/bin/env bash
set -Ee -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
set -a
source "$ROOT/config.env"
set +a
set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/m3pro_hw_ws/install/setup.bash"
[ ! -f "$ROOT/orbbec_ws/install/setup.bash" ] || source "$ROOT/orbbec_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
exec "$ROOT/.venv/bin/python" "$ROOT/tools/manual_arm.py" "$@"
