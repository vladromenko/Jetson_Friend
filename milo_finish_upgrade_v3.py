#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get('JETSON_FRIEND_ROOT', Path.home() / 'Jetson_Friend')).resolve()
PYTHON = ROOT / '.venv' / 'bin' / 'python'


def run(cmd):
    print('+', ' '.join(str(x) for x in cmd), flush=True)
    return subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True, text=True)


def require(path, description):
    if not path.exists():
        raise SystemExit(f'ERROR: {description} missing: {path}')


def replace_once(text, old, new, label):
    if new in text:
        print(f'OK: {label} already applied')
        return text
    if old not in text:
        raise SystemExit(f'ERROR: Could not locate exact block for {label}')
    print(f'OK: applying {label}')
    return text.replace(old, new, 1)


def main():
    print('============================================================')
    print('MILO FINISH UPGRADE V3')
    print('============================================================')

    for rel in ('src/main.py', 'src/ai.py', 'src/emotion.py', 'src/vision.py', 'config.env'):
        require(ROOT / rel, rel)
    require(PYTHON, 'project Python')
    require(ROOT / 'models/vision/emotion.onnx', 'FER+ model')
    require(ROOT / 'models/whisper/ggml-small.en.bin', 'Whisper small.en')
    require(ROOT / 'models/vlm/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf', 'Qwen2.5-VL model')
    require(ROOT / 'models/vlm/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf', 'Qwen2.5-VL projector')

    ai_text = (ROOT / 'src/ai.py').read_text(encoding='utf-8')
    emotion_text = (ROOT / 'src/emotion.py').read_text(encoding='utf-8')
    vision_text = (ROOT / 'src/vision.py').read_text(encoding='utf-8')

    if 'def ask_visual' not in ai_text:
        raise SystemExit('ERROR: first script did not add ai.ask_visual()')
    if 'InteractionMode=emotional_support' not in ai_text:
        raise SystemExit('ERROR: first script did not add emotional support routing')
    if '"contempt"' not in emotion_text:
        raise SystemExit('ERROR: first script did not patch FER+ labels')
    if 'def snapshot_jpeg' not in vision_text:
        raise SystemExit('ERROR: vision.snapshot_jpeg() is missing')

    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = ROOT / f'.milo_upgrade_v3_backup_{stamp}'
    backup.mkdir(parents=True, exist_ok=True)
    for name in ('main.py', 'ai.py', 'emotion.py', 'vision.py'):
        shutil.copy2(ROOT / 'src' / name, backup / name)
    shutil.copy2(ROOT / 'config.env', backup / 'config.env')
    print(f'OK: backup created: {backup}')

    main_path = ROOT / 'src/main.py'
    text = main_path.read_text(encoding='utf-8')

    old_explicit = """                        if (
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
                                    \"last_emotion\"
                                ] = {
                                    \"emotion\": (
                                        explicit_emotion
                                    ),
                                    \"source\": (
                                        \"explicit_speech\"
                                    ),
                                    \"confidence\": 0.95,
                                }

                            debug_print(
                                (
                                    \"Explicit emotion: \"
                                    f\"{explicit_emotion}\"
                                )
                            )"""

    new_explicit = """                        if (
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
                                    \"last_emotion\"
                                ] = {
                                    \"emotion\": (
                                        explicit_emotion
                                    ),
                                    \"source\": (
                                        \"explicit_speech\"
                                    ),
                                    \"confidence\": 0.95,
                                    \"updated_at\": time.time(),
                                }

                            debug_print(
                                (
                                    \"Explicit emotion: \"
                                    f\"{explicit_emotion}\"
                                )
                            )"""

    text = replace_once(text, old_explicit, new_explicit, 'session-level explicit emotion')

    old_reply = """                        emotion_context = (
                            emotion.get_context(
                                person_id=(
                                    person_id
                                )
                            )
                        )

                        if emotion_context:
                            memory_context += (
                                \"\\nTemporary \"
                                \"emotional context: \"
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
                        )"""

    new_reply = """                        emotion_context = (
                            emotion.get_context(
                                person_id=(
                                    person_id
                                )
                            )
                        )

                        if not emotion_context:
                            with state_lock:
                                recent_emotion = state.get(
                                    \"last_emotion\"
                                )

                            if recent_emotion:
                                updated_at = float(
                                    recent_emotion.get(
                                        \"updated_at\",
                                        0.0,
                                    )
                                )

                                if (
                                    updated_at > 0.0
                                    and time.time() - updated_at <= 600.0
                                ):
                                    emotion_context = dict(
                                        recent_emotion
                                    )

                        if emotion_context:
                            memory_context += (
                                \"\\nTemporary \"
                                \"emotional context: \"
                                + str(
                                    emotion_context
                                )
                            )

                        if (
                            looks_visual(text)
                            and not args.no_vision
                        ):
                            image_bytes = vision.snapshot_jpeg(
                                quality=85
                            )

                            reply = ai.ask_visual(
                                text,
                                image_bytes=image_bytes,
                                person_name=name,
                                memory_context=memory_context,
                                emotional_context=emotion_context,
                            )
                        else:
                            reply = ai.ask(
                                text,
                                scene=vision.get_scene(),
                                person_name=name,
                                memory_context=memory_context,
                                emotional_context=emotion_context,
                            )"""

    text = replace_once(text, old_reply, new_reply, 'VLM + emotional-context reply routing')

    old_visual = """                r\"see|seeing|look|looking|camera|face|\"
                r\"wearing|holding|room|around me|\"
                r\"in front of you|what is this|who is here\""""
    new_visual = """                r\"see|seeing|look|looking|camera|face|\"
                r\"wearing|holding|room|around me|surroundings|\"
                r\"describe me|describe what|describe the|\"
                r\"in front of you|what is this|what's this|\"
                r\"what am i holding|what do i have|who is here\""""

    if old_visual in text and 'surroundings|' not in text:
        text = text.replace(old_visual, new_visual, 1)
        print('OK: visual intent expanded')
    else:
        print('OK: visual intent left as-is')

    main_path.write_text(text, encoding='utf-8')

    print('\n== Python validation ==')
    run([PYTHON, '-m', 'py_compile', 'src/main.py', 'src/ai.py', 'src/emotion.py', 'src/vision.py', 'src/speech.py', 'src/identity.py', 'src/memory.py', 'src/behavior.py'])
    run([PYTHON, '-c', "import cv2; cv2.dnn.readNetFromONNX('models/vision/emotion.onnx'); print('FER+ load: OK')"])

    print('\n== TensorRT rebuild ==')
    trtexec = shutil.which('trtexec')
    if trtexec is None:
        for candidate in ('/usr/src/tensorrt/bin/trtexec', '/usr/local/tensorrt/bin/trtexec'):
            if Path(candidate).is_file():
                trtexec = candidate
                break

    if trtexec is None:
        print('WARN: trtexec not found; TensorRT rebuild skipped')
    else:
        builds = (
            ('yolov8n.onnx', 'yolov8n.engine', '/tmp/milo_yolo_trt.log', 1024),
            ('face_detection_yunet.onnx', 'face_detection_yunet.engine', '/tmp/milo_yunet_trt.log', 512),
        )
        for onnx_name, engine_name, logfile, workspace in builds:
            onnx = ROOT / 'models/vision' / onnx_name
            engine = ROOT / 'models/vision' / engine_name
            require(onnx, onnx_name)
            if engine.exists():
                engine.unlink()
            command = [trtexec, f'--onnx={onnx}', f'--saveEngine={engine}', '--fp16', f'--memPoolSize=workspace:{workspace}']
            print('+', ' '.join(command), flush=True)
            with open(logfile, 'w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode == 0 and engine.is_file() and engine.stat().st_size > 0:
                print(f'OK: rebuilt {engine.name}')
            else:
                print(f'WARN: failed to rebuild {engine.name}; log: {logfile}')

    final_main = main_path.read_text(encoding='utf-8')
    checks = {
        'session emotion timestamp': '"updated_at": time.time()' in final_main,
        'visual routing': 'reply = ai.ask_visual(' in final_main,
        'support context passed': 'emotional_context=emotion_context' in final_main,
        'VLM method': 'def ask_visual' in (ROOT / 'src/ai.py').read_text(encoding='utf-8'),
        'FER+ labels': '"contempt"' in (ROOT / 'src/emotion.py').read_text(encoding='utf-8'),
    }

    print('\n== Final checks ==')
    for label, passed in checks.items():
        print(('OK' if passed else 'FAIL') + f': {label}')
    if not all(checks.values()):
        raise SystemExit('ERROR: one or more final checks failed')

    print('\n============================================================')
    print('MILO upgrade v3 completed.')
    print(f'Backup: {backup}')
    print('Next: cd ~/Jetson_Friend && ./start.sh')
    print('============================================================')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f'ERROR: command failed with exit code {exc.returncode}', file=sys.stderr)
        raise SystemExit(exc.returncode)
