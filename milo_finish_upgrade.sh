#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${JETSON_FRIEND_ROOT:-$HOME/Jetson_Friend}"
cd "$ROOT"

log(){ printf '\n============================================================\n== %s\n============================================================\n' "$*"; }
ok(){ printf 'OK: %s\n' "$*"; }
warn(){ printf 'WARN: %s\n' "$*" >&2; }
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -f src/ai.py ] || die "Run this from a Jetson_Friend checkout."
[ -f src/main.py ] || die "src/main.py missing"
[ -f src/emotion.py ] || die "src/emotion.py missing"
[ -f src/model_downloader.py ] || die "src/model_downloader.py missing"
[ -f models/catalog.json ] || die "models/catalog.json missing"
[ -x .venv/bin/python ] || die ".venv/bin/python missing"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/.milo_upgrade_backup_$STAMP"
mkdir -p "$BACKUP"
cp src/ai.py src/main.py src/emotion.py config.env models/catalog.json "$BACKUP"/
ok "Backup created: $BACKUP"

log "1/7 - Model catalog + missing emotion model"

"$ROOT/.venv/bin/python" - <<'PY'
import json
from pathlib import Path

path = Path("models/catalog.json")
data = json.loads(path.read_text(encoding="utf-8"))
models = data.setdefault("models", {})

models["emotion-ferplus-8"] = {
    "name": "FER+ Emotion Recognition 8-class",
    "type": "vision",
    "repo": "onnxmodelzoo/emotion-ferplus-8",
    "notes": "Weak facial-expression cue only; explicit speech has higher priority.",
    "files": [
        {
            "filename": "emotion-ferplus-8.onnx",
            "target": "vision/emotion.onnx"
        }
    ]
}

path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("catalog.json updated")
PY

./.venv/bin/python src/model_downloader.py download emotion-ferplus-8
[ -s models/vision/emotion.onnx ] || die "Emotion model download failed"
ok "FER+ installed"

log "2/7 - Runtime configuration"

"$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path

path = Path("config.env")
text = path.read_text(encoding="utf-8") if path.exists() else ""

root = Path.home() / "Jetson_Friend"
updates = {
    "WHISPER_MODEL": str(root / "models/whisper/ggml-small.en.bin"),
    "LLM_MAX_TOKENS": "180",
    "FACE_EMOTION_MODEL": str(root / "models/vision/emotion.onnx"),
    "EMOTION_INPUT_SIZE": "64",
    "EMOTION_GRAYSCALE": "1",
    "EMOTION_LABELS": "neutral,happy,surprised,sad,angry,disgusted,fearful,contempt",
    "EMOTION_MIN_CONFIDENCE": "0.42",
    "EMOTION_STRONG_CONFIDENCE": "0.70",
    "EMOTION_REQUIRED_OBSERVATIONS": "3",
    "EMOTION_INTERVAL_SEC": "1.25",
    "VLM_ENABLE": "1",
    "VLM_MODEL": str(root / "models/vlm/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"),
    "VLM_MMPROJ": str(root / "models/vlm/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"),
    "VLM_TIMEOUT_SEC": "90",
    "LLM_ENABLE_VISION": "0",
}

lines = text.splitlines()
seen = set()
out = []

for line in lines:
    stripped = line.strip()
    key = stripped.split("=", 1)[0] if "=" in stripped and not stripped.startswith("#") else None
    if key in updates:
        if key not in seen:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
    else:
        out.append(line)

for key, value in updates.items():
    if key not in seen:
        out.append(f"{key}={value}")

path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
print("config.env updated")
PY

[ -s models/whisper/ggml-small.en.bin ] || die "Whisper small.en is missing"
[ -s models/vlm/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf ] || die "Qwen2.5-VL model is missing"
[ -s models/vlm/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf ] || die "Qwen2.5-VL mmproj is missing"
ok "Config updated"

log "3/7 - Patch facial emotion preprocessing + labels"

"$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path

path = Path("src/emotion.py")
text = path.read_text(encoding="utf-8")

old_labels = '''    DEFAULT_LABELS = [
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "sad",
        "surprised",
        "neutral",
    ]'''

new_labels = '''    DEFAULT_LABELS = [
        "neutral",
        "happy",
        "surprised",
        "sad",
        "angry",
        "disgusted",
        "fearful",
        "contempt",
    ]'''

if old_labels in text:
    text = text.replace(old_labels, new_labels, 1)
elif '"contempt"' not in text:
    raise SystemExit("Could not locate DEFAULT_LABELS in src/emotion.py")

old_gray = '''            face = cv2.equalizeHist(
                face
            )

            face = (
                face.astype(
                    np.float32
                )
                / 255.0
            )'''

new_gray = '''            # FER+ expects a raw grayscale 64x64 tensor.
            # Keep pixel values in their original 0..255 range.
            face = face.astype(
                np.float32
            )'''

if old_gray in text:
    text = text.replace(old_gray, new_gray, 1)
elif "FER+ expects a raw grayscale" not in text:
    raise SystemExit("Could not locate grayscale preprocessing block in src/emotion.py")

path.write_text(text, encoding="utf-8")
print("src/emotion.py patched")
PY

./.venv/bin/python -c "import cv2; cv2.dnn.readNetFromONNX('models/vision/emotion.onnx'); print('FER+ OpenCV load: OK')"

log "4/7 - Add explicit emotional support routing"

"$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path

path = Path("src/ai.py")
text = path.read_text(encoding="utf-8")

if "import base64" not in text:
    text = text.replace("import json\n", "import base64\nimport json\n", 1)

old_sig = '''    def ask(
        self,
        user_text,
        scene=None,
        person_name=None,
        memory_context=None,
    ):'''

new_sig = '''    def ask(
        self,
        user_text,
        scene=None,
        person_name=None,
        memory_context=None,
        emotional_context=None,
    ):'''

if old_sig in text:
    text = text.replace(old_sig, new_sig, 1)
elif "emotional_context=None" not in text:
    raise SystemExit("Could not patch ai.ask signature")

anchor = '''        parts.append(
            "User="
            + user_text
        )'''

support = '''        if emotional_context:
            current_emotion = str(
                emotional_context.get(
                    "emotion",
                    "",
                )
            ).strip().lower()

            source = str(
                emotional_context.get(
                    "source",
                    "",
                )
            ).strip()

            support_emotions = {
                "sad",
                "lonely",
                "anxious",
                "stressed",
                "frustrated",
                "afraid",
                "upset",
                "angry",
            }

            if current_emotion in support_emotions:
                parts.append(
                    (
                        "InteractionMode=emotional_support\\n"
                        f"UserEmotion={current_emotion}\\n"
                        f"EmotionSource={source}\\n"
                        "The user needs companionship and meaningful emotional support. "
                        "Respond to the feeling first. Use roughly 3-5 natural spoken "
                        "sentences when useful. Give actual encouragement, comfort, "
                        "gentle optimism, a small distraction, or light humor when "
                        "appropriate. Do not answer with a cold one-liner. Do not jump "
                        "straight into a checklist or generic productivity advice."
                    )
                )
            elif current_emotion:
                parts.append(
                    "CurrentEmotion="
                    + current_emotion
                )

''' + anchor

if anchor in text and "InteractionMode=emotional_support" not in text:
    text = text.replace(anchor, support, 1)
elif "InteractionMode=emotional_support" not in text:
    raise SystemExit("Could not insert emotional support routing")

path.write_text(text, encoding="utf-8")
print("Support routing patched")
PY

log "5/7 - Add live camera -> Qwen2.5-VL routing"

"$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path

path = Path("src/ai.py")
text = path.read_text(encoding="utf-8")

init_anchor = '''        self.mmproj = os.getenv(
            "VLM_MMPROJ",
            "",
        ).strip()
'''

init_insert = init_anchor + '''
        self.vlm_model = os.getenv(
            "VLM_MODEL",
            "",
        ).strip()

        self.vlm_enabled = (
            os.getenv(
                "VLM_ENABLE",
                "1",
            ).strip().lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

        self.vlm_timeout = float(
            os.getenv(
                "VLM_TIMEOUT_SEC",
                "90",
            )
        )
'''

if init_anchor in text and "self.vlm_model =" not in text:
    text = text.replace(init_anchor, init_insert, 1)
elif "self.vlm_model =" not in text:
    raise SystemExit("Could not insert VLM config into ai.py")

method_anchor = '''    def _fallback(
'''

vlm_method = r'''    def ask_visual(
        self,
        user_text,
        image_bytes,
        person_name=None,
        memory_context=None,
        emotional_context=None,
    ):
        if (
            not self.vlm_enabled
            or not self.vlm_model
            or not self.mmproj
            or not os.path.isfile(
                self.vlm_model
            )
            or not os.path.isfile(
                self.mmproj
            )
            or not image_bytes
        ):
            return self.ask(
                user_text,
                person_name=person_name,
                memory_context=memory_context,
                emotional_context=emotional_context,
            )

        encoded = base64.b64encode(
            image_bytes
        ).decode(
            "ascii"
        )

        normal_model = self.model
        normal_mmproj = self.mmproj
        normal_enable_vision = self.enable_vision

        vlm_process = None
        started = time.monotonic()

        try:
            self.stop_server()

            deadline = time.monotonic() + 8.0
            while (
                self._server_alive()
                and time.monotonic() < deadline
            ):
                time.sleep(
                    0.15
                )

            command = [
                self.server_bin,
                "-m",
                self.vlm_model,
                "--mmproj",
                self.mmproj,
                "--host",
                self.server_host,
                "--port",
                str(
                    self.server_port
                ),
                "-c",
                str(
                    max(
                        3072,
                        self.context_size,
                    )
                ),
                "-ngl",
                str(
                    self.gpu_layers
                    if str(
                        self.gpu_layers
                    ).isdigit()
                    else "99"
                ),
                "--flash-attn",
                "on",
                "--parallel",
                "1",
            ]

            print(
                "Starting VLM for live camera request...",
                flush=True,
            )

            vlm_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            deadline = time.monotonic() + 60.0
            while (
                not self._server_alive()
                and time.monotonic() < deadline
            ):
                if vlm_process.poll() is not None:
                    raise RuntimeError(
                        "VLM server exited during startup"
                    )

                time.sleep(
                    0.4
                )

            if not self._server_alive():
                raise RuntimeError(
                    "VLM server did not become ready"
                )

            visual_prompt = (
                "You are MILO looking through your live camera. "
                "Answer the user's visual question from this CURRENT image only. "
                "Describe concrete visible details and spatial relationships. "
                "Do not invent anything that cannot be seen. "
                "If something is uncertain, say so briefly. "
                "For a broad scene-description request, give 3-6 useful spoken "
                "sentences instead of a one-line object list.\n\n"
                "User: "
                + str(
                    user_text
                ).strip()
            )

            if person_name:
                visual_prompt += (
                    "\nKnown person name: "
                    + str(
                        person_name
                    )
                )

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": visual_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/jpeg;base64,"
                                        + encoded
                                    )
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0.35,
                "top_p": 0.85,
                "max_tokens": max(
                    180,
                    self.max_tokens,
                ),
                "stream": False,
            }

            request = urllib.request.Request(
                self.server_url,
                data=json.dumps(
                    payload,
                    separators=(
                        ",",
                        ":",
                    ),
                ).encode(
                    "utf-8"
                ),
                headers={
                    "Content-Type": (
                        "application/json"
                    ),
                },
                method="POST",
            )

            with urllib.request.urlopen(
                request,
                timeout=self.vlm_timeout,
            ) as response:
                data = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

            raw = str(
                data["choices"][0]["message"]["content"]
            ).strip()

            if not raw:
                raw = (
                    "I couldn't get a reliable visual "
                    "description from that frame."
                )

            elapsed = time.monotonic() - started

            print(
                f"VLM response time: {elapsed:.2f}s",
                flush=True,
            )

            return {
                "text": raw,
                "emotion": "neutral",
                "memory": None,
            }

        except Exception as exc:
            print(
                f"VLM error: {exc}",
                flush=True,
            )

            return {
                "text": (
                    "I can see through the camera, but my detailed "
                    "visual model failed on that request."
                ),
                "emotion": "concerned",
                "memory": None,
            }

        finally:
            if (
                vlm_process is not None
                and vlm_process.poll() is None
            ):
                vlm_process.terminate()

                try:
                    vlm_process.wait(
                        timeout=5
                    )
                except subprocess.TimeoutExpired:
                    vlm_process.kill()

            deadline = time.monotonic() + 8.0
            while (
                self._server_alive()
                and time.monotonic() < deadline
            ):
                time.sleep(
                    0.15
                )

            self.model = normal_model
            self.mmproj = normal_mmproj
            self.enable_vision = normal_enable_vision

            try:
                self.ensure_server()
            except Exception as exc:
                print(
                    f"Could not restore normal LLM server: {exc}",
                    flush=True,
                )

'''

if method_anchor in text and "def ask_visual(" not in text:
    text = text.replace(method_anchor, vlm_method + method_anchor, 1)
elif "def ask_visual(" not in text:
    raise SystemExit("Could not insert ask_visual into ai.py")

path.write_text(text, encoding="utf-8")
print("Visual VLM routing patched")
PY

"$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path

path = Path("src/main.py")
text = path.read_text(encoding="utf-8")

old_condition = '''                        if (
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
                                }'''

new_condition = '''                        if (
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
                                }'''

if old_condition in text:
    text = text.replace(old_condition, new_condition, 1)
elif '"updated_at": time.time()' not in text:
    raise SystemExit("Could not patch explicit emotion session state")

old_context = '''                        emotion_context = (
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

new_context = '''                        emotion_context = (
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
                                age = (
                                    time.time()
                                    - float(
                                        recent_emotion.get(
                                            "updated_at",
                                            0.0,
                                        )
                                    )
                                )

                                if age <= 600.0:
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
                                image_bytes=image_bytes,
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

if old_context in text:
    text = text.replace(old_context, new_context, 1)
elif "reply = ai.ask_visual(" not in text:
    raise SystemExit("Could not patch main visual/support routing")

path.write_text(text, encoding="utf-8")
print("src/main.py patched")
PY

log "6/7 - Rebuild TensorRT engines on this exact Jetson"

if command -v trtexec >/dev/null 2>&1; then
    rm -f models/vision/yolov8n.engine models/vision/face_detection_yunet.engine

    if trtexec --help 2>&1 | grep -q -- '--memPoolSize'; then
        trtexec --onnx=models/vision/yolov8n.onnx --saveEngine=models/vision/yolov8n.engine --fp16 --memPoolSize=workspace:1024 >/tmp/milo_yolo_trt.log 2>&1 || warn "YOLO TensorRT rebuild failed; see /tmp/milo_yolo_trt.log"
        trtexec --onnx=models/vision/face_detection_yunet.onnx --saveEngine=models/vision/face_detection_yunet.engine --fp16 --memPoolSize=workspace:512 >/tmp/milo_yunet_trt.log 2>&1 || warn "YuNet TensorRT rebuild failed; see /tmp/milo_yunet_trt.log"
    else
        trtexec --onnx=models/vision/yolov8n.onnx --saveEngine=models/vision/yolov8n.engine --fp16 --workspace=1024 >/tmp/milo_yolo_trt.log 2>&1 || warn "YOLO TensorRT rebuild failed; see /tmp/milo_yolo_trt.log"
        trtexec --onnx=models/vision/face_detection_yunet.onnx --saveEngine=models/vision/face_detection_yunet.engine --fp16 --workspace=512 >/tmp/milo_yunet_trt.log 2>&1 || warn "YuNet TensorRT rebuild failed; see /tmp/milo_yunet_trt.log"
    fi

    [ -s models/vision/yolov8n.engine ] && ok "YOLO engine rebuilt" || warn "YOLO engine unavailable; ONNX fallback should be used"
    [ -s models/vision/face_detection_yunet.engine ] && ok "YuNet engine rebuilt" || warn "YuNet engine unavailable; ONNX fallback should be used"
else
    warn "trtexec not found; skipping TensorRT rebuild"
fi

log "7/7 - Validation"

./.venv/bin/python -m py_compile src/ai.py src/main.py src/emotion.py src/vision.py src/speech.py src/identity.py src/memory.py src/behavior.py

echo
echo "---- MODELS ----"
./.venv/bin/python src/model_downloader.py list

echo
echo "---- CONFIG ----"
grep -E '^(WHISPER_MODEL|LLM_MAX_TOKENS|FACE_EMOTION_MODEL|EMOTION_|VLM_|LLM_ENABLE_VISION)=' config.env || true

echo
echo "---- HARDWARE ----"
v4l2-ctl --list-devices 2>/dev/null || true
arecord -l 2>/dev/null | sed -n '1,14p' || true
aplay -l 2>/dev/null | sed -n '1,14p' || true

echo
echo "============================================================"
echo "MILO upgrade completed."
echo "Backup: $BACKUP"
echo
echo "Next test:"
echo "  cd $ROOT && ./start.sh"
echo
echo "Test phrases:"
echo '  Describe exactly what you see right now.'
echo '  What am I holding?'
echo '  I am feeling very sad. Cheer me up.'
echo "============================================================"
