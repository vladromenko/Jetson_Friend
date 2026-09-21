#!/usr/bin/env bash
set -Eeuo pipefail

packages=(
  build-essential
  cmake
  curl
  ffmpeg
  git
  libsndfile1
  portaudio19-dev
  python3-colcon-common-extensions
  python3-dev
  python3-numpy
  python3-opencv
  python3-pip
  python3-pygame
  python3-rosdep
  python3-scipy
  python3-vcstool
  python3-venv
)

missing=()
for package_name in "${packages[@]}"; do
  dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q 'install ok installed' || missing+=("$package_name")
done

if [ "${#missing[@]}" -eq 0 ]; then
  echo "[READY] system build/audio/vision dependencies"
  exit 0
fi

echo "[INSTALL] system packages: ${missing[*]}"
sudo apt-get update
sudo apt-get install -y "${missing[@]}"
