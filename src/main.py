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
        "JetPack": (
            os.popen(
                "dpkg-query -W -f='${Version}' "
                "nvidia-jetpack 2>/dev/null"
            ).read().strip()
            or "unknown"
        ),
        "CUDA device": (
            "present"
            if os.path.exists(
                "/dev/nvhost-ctrl-gpu"
            )
            else "not detected"
        ),
        "LLM model": ai.model,
        "LLM GPU offload": (
            f"{ai.gpu_layers} layers requested"
        ),
        "Whisper model": speech.whisper_model,
        "Whisper runtime": "CPU (--no-gpu)",
        "LLM backend": "persistent llama-server",
        "TTS voice": speech.voice,
        "camera": f"index {vision.camera_index}",
        "object detector": vision.object_model,
        "face detector": vision.face_model,
    }


def handle_text(
    text,
    ai,
    speech,
    face,
    vision,
    debug,
):
    if not text.strip():
        return

    if debug:
        print(
            f"recognized text: {text}",
            flush=True,
        )

    face.set_state("thinking")
    reply = ai.ask(
        text,
        vision.scene,
    )

    if debug:
        print(
            "detected faces: "
            f"{vision.scene.get('faces', 0)}"
        )
        print(
            "detected objects: "
            f"{', '.join(vision.scene.get('objects', [])) or 'none'}"
        )
        print(
            "vision backend: "
            f"{vision.scene.get('backend', 'unknown')}"
        )
        print(
            f"AI response: {reply['text']}"
        )
        print(
            f"current emotion: {reply['emotion']}",
            flush=True,
        )

    # Show the AI-selected emotion briefly before speech.
    face.set_state(reply["emotion"])
    time.sleep(0.35)

    speech.speak(
        reply["text"],
        face.set_state,
    )

    face.set_state(reply["emotion"])
    time.sleep(1.0)
    face.set_state("neutral")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--no-face",
        action="store_true",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
    )
    parser.add_argument(
        "--no-mic",
        action="store_true",
    )
    args = parser.parse_args()

    face = Face()
    ai = HughAI()
    speech = Speech()
    vision = Vision()

    stop = threading.Event()
    conversation_busy = threading.Event()
    texts = queue.Queue()

    threads = []

    def shutdown(*_):
        if not stop.is_set():
            print(
                "\nStopping Hugh...",
                flush=True,
            )

        stop.set()
        face.stop()
        vision.stop()

    signal.signal(
        signal.SIGINT,
        shutdown,
    )
    signal.signal(
        signal.SIGTERM,
        shutdown,
    )

    if args.debug:
        for key, value in env_status(
            ai,
            speech,
            vision,
        ).items():
            print(
                f"{key}: {value}",
                flush=True,
            )

    if not args.no_face:
        face_thread = threading.Thread(
            target=face.run,
            name="face",
            daemon=False,
        )
        face_thread.start()
        threads.append(face_thread)

    if not args.no_vision:
        vision_thread = threading.Thread(
            target=vision.run,
            args=(face,),
            name="vision",
            daemon=False,
        )
        vision_thread.start()
        threads.append(vision_thread)

    def keyboard():
        while not stop.is_set():
            try:
                line = input(
                    "> " if args.debug else ""
                )
            except EOFError:
                return
            except KeyboardInterrupt:
                shutdown()
                return

            if line.strip():
                texts.put(line)

    def microphone():
        while not stop.is_set():
            blocked = (
                conversation_busy.is_set()
                or speech.is_speaking.is_set()
            )

            if blocked:
                time.sleep(0.05)
            else:
                text = speech.listen_once(
                    face.set_state,
                    stop_event=stop,
                )

                if text and not stop.is_set():
                    conversation_busy.set()
                    texts.put(text)

                if (
                    not conversation_busy.is_set()
                    and not speech.is_speaking.is_set()
                    and not stop.is_set()
                ):
                    face.set_state("neutral")

    keyboard_thread = threading.Thread(
        target=keyboard,
        name="keyboard",
        daemon=True,
    )
    keyboard_thread.start()

    if not args.no_mic:
        mic_thread = threading.Thread(
            target=microphone,
            name="microphone",
            daemon=True,
        )
        mic_thread.start()

    try:
        while not stop.is_set():
            try:
                text = texts.get(
                    timeout=0.2
                )
            except queue.Empty:
                text = None

            if text is not None:
                conversation_busy.set()

                try:
                    handle_text(
                        text,
                        ai,
                        speech,
                        face,
                        vision,
                        args.debug,
                    )
                except Exception as exc:
                    print(
                        f"Conversation error: {exc}",
                        flush=True,
                    )
                    face.set_state("concerned")
                    time.sleep(0.8)
                    face.set_state("neutral")
                finally:
                    conversation_busy.clear()

    finally:
        shutdown()

        deadline = time.monotonic() + 4.0

        for thread in threads:
            remaining = (
                deadline - time.monotonic()
            )

            if remaining > 0:
                thread.join(
                    timeout=remaining
                )

        print(
            "Hugh stopped.",
            flush=True,
        )


if __name__ == "__main__":
    main()
