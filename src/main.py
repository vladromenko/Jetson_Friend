import argparse
import os
import queue
import signal
import threading
import time

from ai import HughAI
from face import Face
from speech import Speech
from vision import Vision


def env_status(ai, speech, vision):
    return {
        "JetPack": os.popen("dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null").read().strip() or "unknown",
        "CUDA": os.popen("nvcc --version 2>/dev/null | grep release | head -1").read().strip() or "unknown",
        "TensorRT": os.popen("trtexec --version 2>/dev/null | head -1").read().strip() or "unknown",
        "GPU active": "yes" if os.path.exists("/dev/nvhost-ctrl-gpu") else "unknown",
        "LLM model": ai.model,
        "LLM GPU offload": f"{ai.gpu_layers} layers requested",
        "Whisper model": speech.whisper_model,
        "Whisper CUDA": "enabled at build",
        "TTS voice": speech.voice,
        "camera": f"index {vision.camera_index}",
        "audio": speech.devices().splitlines()[0] if speech.devices() else "unknown",
        "object detector": vision.object_model,
        "face detector": vision.face_model,
        "vision backend": "OpenCV DNN CUDA if available, ONNX fallback",
    }


def handle_text(text, ai, speech, face, vision, debug):
    if not text.strip():
        return
    if debug:
        print(f"recognized text: {text}", flush=True)
    face.set_state("thinking")
    reply = ai.ask(text, vision.scene)
    if debug:
        print(f"detected faces: {vision.scene.get('faces', 0)}")
        print(f"detected objects: {', '.join(vision.scene.get('objects', [])) or 'none'}")
        print(f"AI response: {reply['text']}")
        print(f"current emotion: {reply['emotion']}", flush=True)
    speech.speak(reply["text"], face.set_state)
    face.set_state(reply["emotion"])
    time.sleep(1.5)
    face.set_state("neutral")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-face", action="store_true")
    args = parser.parse_args()

    face = Face()
    ai = HughAI()
    speech = Speech()
    vision = Vision()
    stop = threading.Event()
    texts = queue.Queue()

    def shutdown(*_):
        stop.set()
        face.running = False
        vision.running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.debug:
        for k, v in env_status(ai, speech, vision).items():
            print(f"{k}: {v}", flush=True)

    if not args.no_face:
        threading.Thread(target=face.run, daemon=True).start()
    threading.Thread(target=vision.run, args=(face,), daemon=True).start()

    def keyboard():
        while not stop.is_set():
            try:
                line = input("> " if args.debug else "")
                texts.put(line)
            except EOFError:
                return

    def microphone():
        while not stop.is_set():
            text = speech.listen_once(face.set_state)
            if text:
                texts.put(text)
            face.set_state("neutral")

    threading.Thread(target=keyboard, daemon=True).start()
    threading.Thread(target=microphone, daemon=True).start()

    while not stop.is_set():
        try:
            handle_text(texts.get(timeout=0.2), ai, speech, face, vision, args.debug)
        except queue.Empty:
            pass


if __name__ == "__main__":
    main()
