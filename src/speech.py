import os
import queue
import subprocess
import tempfile
import wave

import numpy as np
import sounddevice as sd


class Speech:
    def __init__(self):
        self.whisper = os.getenv("WHISPER_BIN", "/app/deps/whisper.cpp/build/bin/whisper-cli")
        self.whisper_model = os.getenv("WHISPER_MODEL", "/app/models/whisper/ggml-base.en.bin")
        self.piper = os.getenv("PIPER_BIN", "piper")
        self.voice = os.getenv("PIPER_VOICE", "/app/models/tts/en_US-ryan-low.onnx")

    def devices(self):
        try:
            return str(sd.query_devices())
        except Exception as exc:
            return f"audio unavailable: {exc}"

    def listen_once(self, set_state, rate=16000):
        q = queue.Queue()
        frames, speaking, silent = [], False, 0

        def cb(indata, *_):
            q.put(indata.copy())

        try:
            with sd.InputStream(channels=1, samplerate=rate, dtype="float32", callback=cb):
                while True:
                    chunk = q.get()
                    rms = float(np.sqrt(np.mean(chunk * chunk)))
                    if rms > 0.018:
                        speaking, silent = True, 0
                        set_state("listening")
                    if speaking:
                        frames.append(chunk)
                        silent = silent + 1 if rms < 0.012 else 0
                    if speaking and silent > 24:
                        break
        except Exception:
            return ""
        if not frames:
            return ""
        audio = np.concatenate(frames, axis=0)
        return self.transcribe(audio, rate)

    def transcribe(self, audio, rate):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes((np.clip(audio[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())
            cmd = [self.whisper, "-m", self.whisper_model, "-f", path, "-l", "en", "-nt", "-np"]
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
            return p.stdout.strip().splitlines()[-1].strip() if p.stdout.strip() else ""
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def speak(self, text, set_state):
        if not text:
            return
        set_state("speaking")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav = f.name
        try:
            synth = subprocess.run([self.piper, "--model", self.voice, "--output_file", wav], input=text, text=True, capture_output=True, timeout=60)
            if synth.returncode == 0:
                subprocess.run(["aplay", "-q", wav], timeout=90)
        except Exception:
            pass
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass
