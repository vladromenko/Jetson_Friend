#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
set -a; source "$ROOT/config.env"; set +a
PYTHON="${APP_PYTHON:-python3}"; [ -x "$PYTHON" ] || PYTHON=python3
printf '=== FILES ===\n'
for f in "$LLAMA_SERVER_BIN" "$LLM_MODEL" "$VLM_MMPROJ" "$WHISPER_BIN" "$WHISPER_MODEL" "$PIPER_BIN" "$PIPER_VOICE" "$FACE_MODEL"; do [ -e "$f" ] && echo "OK $f" || echo "MISSING $f"; done
[ -e "$FACE_ENGINE" ] && echo "OK $FACE_ENGINE" || echo "OPTIONAL MISSING $FACE_ENGINE"
printf '\n=== PYTHON AUDIO/VISION DEPS ===\n'
"$PYTHON" - <<'PY'
for name in ('numpy','sounddevice','scipy','cv2','pygame'):
    try:
        mod=__import__(name); print('OK',name,getattr(mod,'__version__',''))
    except Exception as exc: print('MISSING',name,exc)
PY
printf '\n=== ALSA ===\n'; arecord -l 2>&1 || true; aplay -l 2>&1 || true
printf '\n=== DISPLAY ===\n'; echo "DISPLAY=${DISPLAY:-unset}"; ls -l /tmp/.X11-unix 2>/dev/null || true
printf '\n=== PROCESSES ===\n'; pgrep -af 'micro_ros_agent|orbbec|component_container|llama-server|milo.main' || true
set +u
source /opt/ros/jazzy/setup.bash
[ ! -f "$INTERFACES_WS/install/setup.bash" ] || source "$INTERFACES_WS/install/setup.bash"
[ ! -f "$ORBBEC_WS/install/setup.bash" ] || source "$ORBBEC_WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID
printf '\n=== TOPICS ===\n'; ros2 topic list 2>/dev/null | grep -E 'arm6_feedback|camera/color/image_raw|camera/depth/image_raw' || true
printf '\n=== FEEDBACK ===\n'; timeout 4 ros2 topic echo "$ARM_FEEDBACK_TOPIC" --once || true
printf '\n=== MODEL SERVER ===\n'; curl -fsS "http://${LLM_HOST}:${LLM_PORT}/v1/models" 2>/dev/null | head -c 400 || true; echo
