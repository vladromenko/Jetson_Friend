#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f config.env ]]; then
  echo "config.env missing. Run ./setup.sh first." >&2
  exit 1
fi

if ! docker image inspect jetson-friend:local >/dev/null 2>&1; then
  echo "Docker image jetson-friend:local missing. Run ./setup.sh first." >&2
  exit 1
fi

xhost +local:docker >/dev/null 2>&1 || true
DOCKER=docker
if ! docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"
fi

$DOCKER compose run --rm \
  --service-ports \
  jetson-friend \
  python3 /app/src/main.py "$@"
