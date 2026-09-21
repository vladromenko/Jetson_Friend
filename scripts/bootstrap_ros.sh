#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$ROOT/ros_ws"

source /opt/ros/jazzy/setup.bash
for command_name in vcs rosdep colcon; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "[BLOCKED] missing $command_name; install python3-vcstool python3-rosdep python3-colcon-common-extensions"
    exit 1
  }
done

mkdir -p "$WORKSPACE/src"
vcs import "$WORKSPACE/src" --recursive --skip-existing < "$WORKSPACE/dependencies.repos"

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update
rosdep install --from-paths "$WORKSPACE/src" --ignore-src --rosdistro jazzy -r -y

cd "$WORKSPACE"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
colcon build --symlink-install --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release
echo "[READY] camera, arm messages and micro-ROS agent built in $WORKSPACE"
