#!/usr/bin/env bash
set -Eeuo pipefail
ROTATION="${1:-normal}"
export DISPLAY="${DISPLAY:-:0}"
OUTPUT="${MILO_DISPLAY_OUTPUT:-DP-1}"
case "$ROTATION" in
  normal|left|right|inverted) ;;
  *) echo "usage: $0 [normal|left|right|inverted]"; exit 2 ;;
esac
if [ "$ROTATION" = normal ] || [ "$ROTATION" = inverted ]; then
  MODE="1920x1080"
else
  MODE="1080x1920"
fi
if xrandr --query | grep -q "^$OUTPUT connected"; then
  if xrandr --query | grep -A8 "^$OUTPUT connected" | grep -q "$MODE"; then
    xrandr --output "$OUTPUT" --primary --mode "$MODE" --rotate "$ROTATION"
  else
    xrandr --output "$OUTPUT" --primary --auto --rotate "$ROTATION"
  fi
else
  echo "[DISPLAY] output $OUTPUT is not connected"
  xrandr --query
  exit 1
fi
xrandr --query | sed -n "1,80p"
