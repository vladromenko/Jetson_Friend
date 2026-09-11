#!/usr/bin/env python3
"""Synthetic utterance -> actual STT -> streaming LLM -> TTS -> speaker, with vision.

The start is an already captured audio buffer, not an acoustic end-of-utterance.
No user audio, frames, identities or memories are recorded.
"""

import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ai import HughAI
from speech import Speech
from telemetry import TurnTiming
from vision import Vision


class Face:
    def set_state(self, *_):
        pass

    def set_gaze(self, *_):
        pass


def main():
    log = open("artifacts/whole_turn.tegrastats", "w")
    stats = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=log)
    vision, speech, face = Vision(), Speech(), Face()
    worker = threading.Thread(target=vision.run, args=(face,), daemon=True)
    worker.start()
    ai = HughAI()
    speech.warmup()
    rows = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "input.wav"
            speech.synthesize("Hello Milo. How are you?", wav)
            audio, rate = sf.read(wav, dtype="float32", always_2d=True)
            for _ in range(2):
                timing = TurnTiming(source="synthetic_audio_buffer")
                text = speech.transcribe(audio, rate)
                timing.mark("stt_complete")
                # No profile is selected for this synthetic turn.
                timing.mark("memory_complete")
                phrases = queue.Queue()
                errors = []
                answer = []

                def generate():
                    try:
                        for phrase in ai.stream_reply(text, timing=timing):
                            phrases.put(phrase)
                    except Exception as exc:
                        errors.append(str(exc))
                    finally:
                        phrases.put(None)

                producer = threading.Thread(target=generate)
                producer.start()
                while True:
                    phrase = phrases.get(timeout=90)
                    if phrase is None:
                        break
                    answer.append(phrase)
                    speech.speak(phrase, face.set_state, timing=timing)
                producer.join()
                result = timing.finish()
                result.update(errors=errors, answer=answer, transcript=text)
                result["vision"] = vision.get_scene().get("metrics")
                rows.append(result)
    finally:
        vision.stop()
        worker.join(timeout=5)
        ai.close()
        stats.terminate()
        stats.wait(timeout=5)
        log.close()
        Path("artifacts/whole_turn.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
