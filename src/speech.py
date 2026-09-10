import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly


class Speech:
    def __init__(self):
        self.whisper = os.getenv(
            "WHISPER_BIN",
            "/app/deps/whisper.cpp/build/bin/whisper-cli",
        )
        self.whisper_model = os.getenv(
            "WHISPER_MODEL",
            "/app/models/whisper/ggml-base.en.bin",
        )

        configured_piper = os.getenv("PIPER_BIN", "piper")
        if os.path.isfile(configured_piper):
            self.piper = configured_piper
        else:
            self.piper = shutil.which("piper") or configured_piper

        self.voice = os.getenv(
            "PIPER_VOICE",
            "/app/models/tts/en_US-ryan-low.onnx",
        )
        self.input_device_name = os.getenv("AUDIO_INPUT_DEVICE", "UM02")
        self.output_hint = os.getenv("AUDIO_OUTPUT_DEVICE", "UACDemo")

        self.is_speaking = threading.Event()
        self._last_mic_error = ""
        self._last_mic_error_time = 0.0

    def devices(self):
        try:
            return str(sd.query_devices())
        except Exception as exc:
            return f"audio unavailable: {exc}"

    def _find_input_device(self):
        try:
            devices = sd.query_devices()
        except Exception as exc:
            self._print_mic_error(f"cannot enumerate audio devices: {exc}")
            return None

        wanted = self.input_device_name.strip().lower()

        for index, device in enumerate(devices):
            name = str(device.get("name", ""))
            inputs = int(device.get("max_input_channels", 0))

            if inputs > 0 and wanted in name.lower():
                return index

        self._print_mic_error(
            f"input device '{self.input_device_name}' is not available"
        )
        return None

    def _find_output_device(self):
        configured = self.output_hint.strip()

        if configured.startswith(("hw:", "plughw:", "default")):
            return configured

        try:
            process = subprocess.run(
                ["aplay", "-l"],
                text=True,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            return "default"

        wanted = configured.lower()

        for line in process.stdout.splitlines():
            lower = line.lower()

            if "card " in lower and wanted in lower:
                match = re.search(r"card\s+(\d+):", line, flags=re.I)

                if match:
                    card = match.group(1)
                    return f"plughw:{card},0"

        # Useful fallback for the USB speaker previously used by Hugh.
        for line in process.stdout.splitlines():
            lower = line.lower()

            if "card " in lower and (
                "uacdemo" in lower
                or "usb audio" in lower
            ):
                match = re.search(r"card\s+(\d+):", line, flags=re.I)

                if match:
                    return f"plughw:{match.group(1)},0"

        return "default"

    def _print_mic_error(self, message):
        now = time.monotonic()

        if (
            message != self._last_mic_error
            or now - self._last_mic_error_time >= 5.0
        ):
            print(f"Microphone error: {message}", flush=True)
            self._last_mic_error = message
            self._last_mic_error_time = now

    def listen_once(
        self,
        set_state,
        stop_event=None,
        rate=44100,
        max_seconds=10.0,
    ):
        if self.is_speaking.is_set():
            return ""

        input_device = self._find_input_device()

        if input_device is None:
            time.sleep(0.5)
            return ""

        audio_queue = queue.Queue()
        frames = []
        speaking = False
        silence_blocks = 0
        started_at = time.monotonic()

        speech_threshold = 0.006
        silence_threshold = 0.004
        blocksize = 1024
        silence_needed = 14

        def callback(indata, frames_count, time_info, status):
            audio_queue.put(indata.copy())

        try:
            with sd.InputStream(
                device=input_device,
                channels=1,
                samplerate=rate,
                blocksize=blocksize,
                dtype="float32",
                callback=callback,
            ):
                active = True

                while active:
                    stop_requested = (
                        stop_event is not None
                        and stop_event.is_set()
                    )
                    timed_out = (
                        time.monotonic() - started_at >= max_seconds
                    )

                    if (
                        stop_requested
                        or timed_out
                        or self.is_speaking.is_set()
                    ):
                        active = False
                    else:
                        try:
                            chunk = audio_queue.get(timeout=0.25)
                        except queue.Empty:
                            chunk = None

                        if chunk is not None:
                            rms = float(np.sqrt(np.mean(chunk * chunk)))

                            if rms > speech_threshold:
                                speaking = True
                                silence_blocks = 0
                                set_state("listening")

                            if speaking:
                                frames.append(chunk)

                                if rms < silence_threshold:
                                    silence_blocks += 1
                                else:
                                    silence_blocks = 0

                                if silence_blocks >= silence_needed:
                                    active = False

        except Exception as exc:
            self._print_mic_error(str(exc))
            return ""

        if not frames:
            return ""

        audio = np.concatenate(frames, axis=0)
        audio_16k = resample_poly(
            audio[:, 0],
            16000,
            rate,
        ).astype(np.float32)
        audio_16k = audio_16k.reshape(-1, 1)

        return self.transcribe(audio_16k, 16000)

    def transcribe(self, audio, rate):
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as file:
            path = file.name

        try:
            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(rate)

                pcm = (
                    np.clip(audio[:, 0], -1, 1) * 32767
                ).astype(np.int16)
                wav_file.writeframes(pcm.tobytes())

            command = [
                self.whisper,
                "-m",
                self.whisper_model,
                "-f",
                path,
                "-l",
                "en",
                "-nt",
                "-np",
            ]

            process = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=120,
            )

            if process.returncode != 0:
                print(
                    f"Whisper error: {process.stderr.strip()}",
                    flush=True,
                )
                return ""

            text = process.stdout.strip()

            if not text:
                return ""

            recognized = text.splitlines()[-1].strip()

            normalized = recognized.strip().lower()

            ignored = {
                "",
                "[blank_audio]",
                "[blank audio]",
                "[silence]",
                "[no speech]",
                "[music]",
                "(music)",
                "(sighs)",
                "[sighs]",
                "(sigh)",
                "[sigh]",
                "(breathing)",
                "[breathing]",
                "(background noise)",
                "[background noise]",
            }

            if normalized in ignored:
                return ""

            if normalized.startswith("[") and normalized.endswith("]"):
                return ""

            if normalized.startswith("(") and normalized.endswith(")"):
                return ""

            return recognized

        except Exception as exc:
            print(f"Whisper error: {exc}", flush=True)
            return ""

        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def speak(self, text, set_state):
        if not text:
            return

        self.is_speaking.set()
        set_state("speaking")

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as file:
            wav_path = file.name

        try:
            synth = subprocess.run(
                [
                    self.piper,
                    "--model",
                    self.voice,
                    "--output_file",
                    wav_path,
                ],
                input=text,
                text=True,
                capture_output=True,
                timeout=60,
            )

            if synth.returncode != 0:
                print(
                    f"Piper error: {synth.stderr.strip()}",
                    flush=True,
                )
                return

            output_device = self._find_output_device()
            print(
                f"Audio output: {output_device}",
                flush=True,
            )

            playback = subprocess.run(
                [
                    "pw-play",
                    wav_path,
                ],
                text=True,
                capture_output=True,
                timeout=90,
            )

            if playback.returncode != 0:
                print(
                    f"Speaker error: {playback.stderr.strip()}",
                    flush=True,
                )

        except Exception as exc:
            print(f"Speaker error: {exc}", flush=True)

        finally:
            self.is_speaking.clear()

            try:
                os.unlink(wav_path)
            except OSError:
                pass
