#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${JETSON_FRIEND_ROOT:-$HOME/Jetson_Friend}"
cd "$ROOT"

log(){ printf '\n============================================================\n== %s\n============================================================\n' "$*"; }
ok(){ printf 'OK: %s\n' "$*"; }
warn(){ printf 'WARN: %s\n' "$*" >&2; }
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -f src/ai.py ] || die "src/ai.py missing"
[ -f src/main.py ] || die "src/main.py missing"
[ -f src/emotion.py ] || die "src/emotion.py missing"
[ -f src/vision.py ] || die "src/vision.py missing"
[ -f config.env ] || die "config.env missing"
[ -x .venv/bin/python ] || die ".venv/bin/python missing"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/.milo_upgrade_v2_backup_$STAMP"
mkdir -p "$BACKUP"
cp src/ai.py src/main.py src/emotion.py config.env "$BACKUP"/
ok "Backup created: $BACKUP"

log "1/5 - Verify models and current partial upgrade"

[ -s models/whisper/ggml-small.en.bin ] || die "Whisper small.en missing"
[ -s models/vision/emotion.onnx ] || die "Emotion model missing"
[ -s models/vlm/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf ] || die "VLM model missing"
[ -s models/vlm/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf ] || die "VLM mmproj missing"

grep -q "InteractionMode=emotional_support" src/ai.py || die "ai.py support patch from first script is missing"
grep -q "def ask_visual" src/ai.py || die "ai.py VLM patch from first script is missing"
grep -q '"contempt"' src/emotion.py || die "emotion.py FER+ patch from first script is missing"

ok "Models and partial patches are present"

log "2/5 - Patch main.py for session emotion + real VLM routing"

"$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path

path = Path("src/main.py")
text = path.read_text(encoding="utf-8")

old_explicit = '''                        if (
                            explicit_emotion
                            and person_id
                            and not args.no_emotion
                        ):
                            emotion.set_explicit_emotion(
                                person_id=person_id,
                                emotion=explicit_emotion,
                                text=text,
                                confidence=0.95,
                            )

                            with state_lock:
                                state[
                                    "last_emotion"
                                ] = {
                                    "emotion": (
                                        explicit_emotion
                                    ),
                                    "source": (
                                        "explicit_speech"
                                    ),
                                    "confidence": 0.95,
                                }

                            debug_print(
                                (
                                    "Explicit emotion: "
                                    f"{explicit_emotion}"
                                )
                            )'''

new_explicit = '''                        if (
                            explicit_emotion
                            and not args.no_emotion
                        ):
                            if person_id:
                                emotion.set_explicit_emotion(
                                    person_id=person_id,
                                    emotion=explicit_emotion,
                                    text=text,
                                    confidence=0.95,
                                )

                            with state_lock:
                                state[
                                    "last_emotion"
                                ] = {
                                    "emotion": (
                                        explicit_emotion
                                    ),
                                    "source": (
                                        "explicit_speech"
                                    ),
                                    "confidence": 0.95,
                                    "updated_at": time.time(),
                                }

                            debug_print(
                                (
                                    "Explicit emotion: "
                                    f"{explicit_emotion}"
                                )
                            )'''

if old_explicit in text:
    text = text.replace(
        old_explicit,
        new_explicit,
        1,
    )
elif '"updated_at": time.time()' not in text:
    raise SystemExit(
        "Could not patch explicit-emotion block in main.py"
    )

old_reply = '''                        emotion_context = (
                            emotion.get_context(
                                person_id=(
                                    person_id
                                )
                            )
                        )

                        if emotion_context:
                            memory_context += (
                                "\nTemporary "
                                "emotional context: "
                                + str(
                                    emotion_context
                                )
                            )

                        reply = ai.ask(
                            text,
                            scene=(
                                vision.get_scene()
                            ),
                            person_name=name,
                            memory_context=(
                                memory_context
                            ),
                        )'''

new_reply = '''                        emotion_context = (
                            emotion.get_context(
                                person_id=(
                                    person_id
                                )
                            )
                        )

                        if not emotion_context:
                            with state_lock:
                                recent_emotion = (
                                    state.get(
                                        "last_emotion"
                                    )
                                )

                            if recent_emotion:
                                updated_at = float(
                                    recent_emotion.get(
                                        "updated_at",
                                        0.0,
                                    )
                                )

                                if (
                                    updated_at > 0.0
                                    and time.time()
                                    - updated_at
                                    <= 600.0
                                ):
                                    emotion_context = dict(
                                        recent_emotion
                                    )

                        if emotion_context:
                            memory_context += (
                                "\nTemporary "
                                "emotional context: "
                                + str(
                                    emotion_context
                                )
                            )

                        if (
                            looks_visual(
                                text
                            )
                            and not args.no_vision
                        ):
                            image_bytes = (
                                vision.snapshot_jpeg(
                                    quality=85
                                )
                            )

                            reply = ai.ask_visual(
                                text,
                                image_bytes=(
                                    image_bytes
                                ),
                                person_name=name,
                                memory_context=(
                                    memory_context
                                ),
                                emotional_context=(
                                    emotion_context
                                ),
                            )

                        else:
                            reply = ai.ask(
                                text,
                                scene=(
                                    vision.get_scene()
                                ),
                                person_name=name,
                                memory_context=(
                                    memory_context
                                ),
                                emotional_context=(
                                    emotion_context
                                ),
                            )'''

if old_reply in text:
    text = text.replace(
        old_reply,
        new_reply,
        1,
    )
elif "reply = ai.ask_visual(" not in text:
    raise SystemExit(
        "Could not patch VLM/support reply block in main.py"
    )

path.write_text(
    text,
    encoding="utf-8",
)

print("src/main.py patched")
PY

log "3/5 - Make visual intent a little broader"

"$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path

path = Path("src/main.py")
text = path.read_text(encoding="utf-8")

old = '''                r"see|seeing|look|looking|camera|face|"
                r"wearing|holding|room|around me|"
                r"in front of you|what is this|who is here"'''

new = '''                r"see|seeing|look|looking|camera|face|"
                r"wearing|holding|room|around me|surroundings|"
                r"describe me|describe what|describe the|"
                r"in front of you|what is this|what's this|"
                r"what am i holding|what do i have|who is here"'''

if old in text:
    text = text.replace(
        old,
        new,
        1,
    )

path.write_text(
    text,
    encoding="utf-8",
)

print("Visual intent rules updated")
PY

log "4/5 - Rebuild TensorRT engines locally"

if command -v trtexec >/dev/null 2>&1; then
    rm -f         models/vision/yolov8n.engine         models/vision/face_detection_yunet.engine

    if trtexec --help 2>&1 | grep -q -- '--memPoolSize'; then
        trtexec             --onnx=models/vision/yolov8n.onnx             --saveEngine=models/vision/yolov8n.engine             --fp16             --memPoolSize=workspace:1024             >/tmp/milo_yolo_trt.log 2>&1             || warn "YOLO TensorRT build failed; see /tmp/milo_yolo_trt.log"

        trtexec             --onnx=models/vision/face_detection_yunet.onnx             --saveEngine=models/vision/face_detection_yunet.engine             --fp16             --memPoolSize=workspace:512             >/tmp/milo_yunet_trt.log 2>&1             || warn "YuNet TensorRT build failed; see /tmp/milo_yunet_trt.log"
    else
        trtexec             --onnx=models/vision/yolov8n.onnx             --saveEngine=models/vision/yolov8n.engine             --fp16             --workspace=1024             >/tmp/milo_yolo_trt.log 2>&1             || warn "YOLO TensorRT build failed; see /tmp/milo_yolo_trt.log"

        trtexec             --onnx=models/vision/face_detection_yunet.onnx             --saveEngine=models/vision/face_detection_yunet.engine             --fp16             --workspace=512             >/tmp/milo_yunet_trt.log 2>&1             || warn "YuNet TensorRT build failed; see /tmp/milo_yunet_trt.log"
    fi

    if [ -s models/vision/yolov8n.engine ]; then
        ok "YOLO TensorRT engine rebuilt"
    else
        warn "YOLO engine missing after build"
    fi

    if [ -s models/vision/face_detection_yunet.engine ]; then
        ok "YuNet TensorRT engine rebuilt"
    else
        warn "YuNet engine missing after build"
    fi
else
    warn "trtexec not found; TensorRT rebuild skipped"
fi

log "5/5 - Validate Python and config"

./.venv/bin/python -m py_compile     src/ai.py     src/main.py     src/emotion.py     src/vision.py     src/speech.py     src/identity.py     src/memory.py     src/behavior.py

./.venv/bin/python -c "import cv2; cv2.dnn.readNetFromONNX('models/vision/emotion.onnx'); print('FER+ load: OK')"

echo
echo "---- IMPORTANT CONFIG ----"
grep -E '^(WHISPER_MODEL|LLM_MAX_TOKENS|FACE_EMOTION_MODEL|EMOTION_|VLM_|LLM_ENABLE_VISION)=' config.env || true

echo
echo "---- PATCH CHECK ----"
grep -n 'InteractionMode=emotional_support' src/ai.py || true
grep -n 'def ask_visual' src/ai.py || true
grep -n 'reply = ai.ask_visual' src/main.py || true
grep -n '"updated_at": time.time()' src/main.py || true

echo
echo "============================================================"
echo "MILO upgrade v2 completed."
echo "Backup: $BACKUP"
echo
echo "Now run:"
echo "  cd $ROOT && ./start.sh"
echo
echo "Test:"
echo '  1. Describe exactly what you see right now.'
echo '  2. What am I holding?'
echo '  3. I am feeling very sad. Cheer me up.'
echo "============================================================"
