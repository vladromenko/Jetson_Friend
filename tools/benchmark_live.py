#!/usr/bin/env python3
"""Hardware smoke/latency run. Frames stay on Jetson and are never written to disk."""

import json
from pathlib import Path
import queue
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ai import HughAI
from speech import Speech
from telemetry import TurnTiming
from vision import Vision


class DisplayState:
    def set_gaze(self, *args):
        pass

    def set_state(self, *args):
        pass


def main():
    vision = Vision()
    face = DisplayState()
    worker = threading.Thread(target=vision.run, args=(face,), daemon=True)
    worker.start()
    ai = HughAI()
    speech = Speech()
    speech.warmup()
    rows = []
    try:
        for text in [
            "Hello Milo. How are you?",
            "I have a robotics presentation tomorrow and feel nervous.",
        ]:
            timing = TurnTiming(source="synthetic_text")
            q = queue.Queue()
            errors = []

            def generate():
                try:
                    for phrase in ai.stream_reply(text, timing=timing):
                        q.put(phrase)
                except Exception as exc:
                    errors.append(str(exc))
                finally:
                    q.put(None)

            producer = threading.Thread(target=generate)
            producer.start()
            answer = []
            while True:
                phrase = q.get(timeout=90)
                if phrase is None:
                    break
                answer.append(phrase)
                speech.speak(phrase, face.set_state, timing=timing)
            producer.join()
            row = timing.finish()
            row.update(answer=answer, errors=errors)
            rows.append(row)
        scene = vision.get_scene()
        print(
            json.dumps(
                {
                    "turns": rows,
                    "vision": {
                        k: scene.get(k)
                        for k in ("vision_backend", "faces", "updated", "motion")
                    },
                },
                indent=2,
            )
        )
        Path("artifacts/live_stream.json").write_text(
            json.dumps(
                {"turns": rows, "vision_backend": scene.get("vision_backend")}, indent=2
            )
        )
    finally:
        vision.stop()
        worker.join(timeout=5)
        ai.close()


if __name__ == "__main__":
    main()
