import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from collections import deque

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly


class Speech:
    def __init__(self):
        root = os.getenv(
            "JETSON_FRIEND_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        self.whisper = os.getenv(
            "WHISPER_BIN",
            os.path.join(
                root,
                "deps",
                "whisper.cpp",
                "build",
                "bin",
                "whisper-cli",
            ),
        )

        self.whisper_model = os.getenv(
            "WHISPER_MODEL",
            os.path.join(
                root,
                "models",
                "whisper",
                "ggml-base.en.bin",
            ),
        )

        configured_piper = os.getenv(
            "PIPER_BIN",
            "piper",
        )

        if os.path.isfile(configured_piper):
            self.piper = configured_piper
        else:
            self.piper = shutil.which("piper") or configured_piper

        self.voice = os.getenv(
            "PIPER_VOICE",
            os.path.join(
                root,
                "models",
                "tts",
                "en_US-ryan-low.onnx",
            ),
        )

        self.input_device_name = os.getenv(
            "AUDIO_INPUT_DEVICE",
            "UM02",
        )

        self.output_hint = os.getenv(
            "AUDIO_OUTPUT_DEVICE",
            "UACDemo",
        )

        self.sample_rate = int(
            os.getenv(
                "MIC_SAMPLE_RATE",
                "44100",
            )
        )

        self.blocksize = int(
            os.getenv(
                "MIC_BLOCK_SIZE",
                "1024",
            )
        )

        self.speech_threshold = float(
            os.getenv(
                "VAD_SPEECH_THRESHOLD",
                "0.006",
            )
        )

        self.silence_threshold = float(
            os.getenv(
                "VAD_SILENCE_THRESHOLD",
                "0.004",
            )
        )

        self.end_silence_seconds = float(
            os.getenv(
                "VAD_END_SILENCE_SEC",
                "0.38",
            )
        )

        self.min_speech_seconds = float(
            os.getenv(
                "VAD_MIN_SPEECH_SEC",
                "0.25",
            )
        )

        self.pre_roll_seconds = float(
            os.getenv(
                "VAD_PRE_ROLL_SEC",
                "0.20",
            )
        )

        self.max_seconds = float(
            os.getenv(
                "MIC_MAX_SECONDS",
                "12",
            )
        )

        self.minimum_audio_rms = float(
            os.getenv(
                "WHISPER_MIN_AUDIO_RMS",
                "0.0025",
            )
        )

        self._voice_instance = None
        self._voice_failed = False
        self._synthesis_lock = threading.RLock()
        self.is_speaking = threading.Event()

        self._input_device = None
        self._output_device = None

        self._last_mic_error = ""
        self._last_mic_error_time = 0.0

        self._playback_lock = threading.RLock()
        self._playback_process = None

        self._active_face = None

    def devices(self):
        try:
            return str(sd.query_devices())

        except Exception as exc:
            return f"audio unavailable: {exc}"

    def _get_face_from_callback(
        self,
        set_state,
    ):
        """
        main.py passes face.set_state.

        A bound method contains a reference to
        its Face instance in __self__, so we can
        access set_speaking_active() without
        changing main.py.
        """

        owner = getattr(
            set_state,
            "__self__",
            None,
        )

        if owner is not None and hasattr(
            owner,
            "set_speaking_active",
        ):
            return owner

        return None

    def _set_mouth_active(
        self,
        active,
    ):
        face = self._active_face

        if face is not None and hasattr(
            face,
            "set_speaking_active",
        ):
            try:
                face.set_speaking_active(bool(active))

            except Exception:
                pass

    def _find_input_device(
        self,
        refresh=False,
    ):
        if self._input_device is not None and not refresh:
            return self._input_device

        try:
            devices = sd.query_devices()

        except Exception as exc:
            self._print_mic_error((f"cannot enumerate audio devices: {exc}"))

            return None

        wanted = self.input_device_name.strip().lower()

        found = None

        if wanted:
            for index, device in enumerate(devices):
                name = str(
                    device.get(
                        "name",
                        "",
                    )
                )

                inputs = int(
                    device.get(
                        "max_input_channels",
                        0,
                    )
                )

                if found is None and inputs > 0 and wanted in name.lower():
                    found = index

        if found is not None:
            self._input_device = found

            print(
                (f"Microphone: {devices[found].get('name')}"),
                flush=True,
            )

            return found

        try:
            default_input = int(sd.default.device[0])

            if default_input >= 0:
                device = devices[default_input]

                if (
                    int(
                        device.get(
                            "max_input_channels",
                            0,
                        )
                    )
                    > 0
                ):
                    self._input_device = default_input

                    print(
                        (f"Using default microphone: {device.get('name')}"),
                        flush=True,
                    )

                    return default_input

        except Exception:
            pass

        self._print_mic_error(
            (f"input device '{self.input_device_name}' is not available")
        )

        return None

    def _find_output_device(
        self,
        refresh=False,
    ):
        if self._output_device is not None and not refresh:
            return self._output_device

        configured = self.output_hint.strip()

        if configured.startswith(
            (
                "hw:",
                "plughw:",
                "default",
                "pipewire",
            )
        ):
            self._output_device = configured
            return configured

        # Desktop audio already owns the ALSA device. Use its shared sink
        # instead of competing for exclusive hardware access.
        if shutil.which("pw-play") and shutil.which("pactl"):
            try:
                sinks = subprocess.run(
                    ["pactl", "list", "short", "sinks"],
                    text=True,
                    capture_output=True,
                    timeout=5,
                )
                if sinks.returncode == 0 and configured:
                    for line in sinks.stdout.splitlines():
                        fields = line.split()
                        if len(fields) >= 2 and configured.lower() in fields[1].lower():
                            self._output_device = f"pipewire:{fields[1]}"
                            return self._output_device
            except (OSError, subprocess.TimeoutExpired):
                pass

        try:
            process = subprocess.run(
                [
                    "aplay",
                    "-l",
                ],
                text=True,
                capture_output=True,
                timeout=5,
            )

        except Exception:
            self._output_device = "default"
            return "default"

        wanted = configured.lower()

        selected = None

        for line in process.stdout.splitlines():
            lower = line.lower()

            if selected is None and "card " in lower and wanted and wanted in lower:
                match = re.search(
                    r"card\s+(\d+):",
                    line,
                    flags=re.I,
                )

                if match:
                    selected = f"plughw:{match.group(1)},0"

        if selected is None:
            for line in process.stdout.splitlines():
                lower = line.lower()

                if (
                    selected is None
                    and "card " in lower
                    and ("uacdemo" in lower or "usb audio" in lower)
                ):
                    match = re.search(
                        r"card\s+(\d+):",
                        line,
                        flags=re.I,
                    )

                    if match:
                        selected = f"plughw:{match.group(1)},0"

        if selected is None:
            selected = "default"

        self._output_device = selected

        return selected

    def _print_mic_error(
        self,
        message,
    ):
        now = time.monotonic()

        should_print = message != self._last_mic_error or (
            now - self._last_mic_error_time >= 5.0
        )

        if should_print:
            print(
                (f"Microphone error: {message}"),
                flush=True,
            )

            self._last_mic_error = message
            self._last_mic_error_time = now

    @staticmethod
    def _rms(
        chunk,
    ):
        if chunk is None or chunk.size == 0:
            return 0.0

        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        chunk,
                        dtype=np.float32,
                    )
                )
            )
        )

    def _capture_rate(self, device, requested):
        cached = getattr(self, "_validated_capture", None)
        if cached and cached[:2] == (device, requested):
            return cached[2]
        native = int(sd.query_devices(device, "input")["default_samplerate"])
        for rate in dict.fromkeys((requested, 16000, native)):
            try:
                sd.check_input_settings(
                    device=device, channels=1, dtype="float32", samplerate=rate
                )
                self._validated_capture = (device, requested, rate)
                return rate
            except (sd.PortAudioError, ValueError):
                continue
        raise RuntimeError("Microphone has no supported mono capture format")

    def listen_once(
        self,
        set_state,
        stop_event=None,
        rate=None,
        max_seconds=None,
    ):
        if self.is_speaking.is_set():
            return ""

        rate = int(rate) if rate else self.sample_rate

        max_seconds = float(max_seconds) if max_seconds else self.max_seconds

        input_device = self._find_input_device()

        if input_device is None:
            time.sleep(0.5)
            return ""

        try:
            rate = self._capture_rate(input_device, rate)
        except Exception as exc:
            self._print_mic_error(str(exc))
            if stop_event is not None:
                stop_event.wait(1.0)
            else:
                time.sleep(1.0)
            return ""

        audio_queue = queue.Queue(maxsize=64)

        frames = []

        pre_roll_blocks = max(
            1,
            int(self.pre_roll_seconds * rate / self.blocksize),
        )

        pre_roll = deque(maxlen=pre_roll_blocks)

        silence_needed = max(
            2,
            int(self.end_silence_seconds * rate / self.blocksize),
        )

        minimum_speech_blocks = max(
            1,
            int(self.min_speech_seconds * rate / self.blocksize),
        )

        speaking = False
        speech_blocks = 0
        silence_blocks = 0
        finished = False

        started_at = time.monotonic()

        def callback(
            indata,
            frames_count,
            time_info,
            status,
        ):
            try:
                audio_queue.put_nowait(indata.copy())

            except queue.Full:
                try:
                    audio_queue.get_nowait()

                except queue.Empty:
                    pass

                try:
                    audio_queue.put_nowait(indata.copy())

                except queue.Full:
                    pass

        try:
            with sd.InputStream(
                device=input_device,
                channels=1,
                samplerate=rate,
                blocksize=self.blocksize,
                dtype="float32",
                callback=callback,
            ):
                while not finished:
                    stop_requested = stop_event is not None and stop_event.is_set()

                    if stop_requested:
                        return ""

                    if self.is_speaking.is_set():
                        return ""

                    elapsed = time.monotonic() - started_at

                    if elapsed >= max_seconds:
                        finished = True

                    else:
                        chunk = None

                        try:
                            chunk = audio_queue.get(timeout=0.20)

                        except queue.Empty:
                            chunk = None

                        if chunk is not None:
                            rms = self._rms(chunk)

                            if not speaking:
                                pre_roll.append(chunk)

                                if rms >= self.speech_threshold:
                                    speaking = True

                                    frames.extend(list(pre_roll))

                                    pre_roll.clear()

                                    speech_blocks = 1
                                    silence_blocks = 0

                                    set_state("listening")

                            else:
                                frames.append(chunk)

                                if rms >= self.silence_threshold:
                                    speech_blocks += 1
                                    silence_blocks = 0

                                else:
                                    silence_blocks += 1

                                enough_silence = silence_blocks >= silence_needed

                                enough_speech = speech_blocks >= minimum_speech_blocks

                                if enough_silence and enough_speech:
                                    finished = True

        except Exception as exc:
            self._input_device = None
            self._validated_capture = None
            self._print_mic_error(str(exc))
            if stop_event is not None:
                stop_event.wait(1.0)
            else:
                time.sleep(1.0)
            return ""

        if not speaking or not frames or speech_blocks < minimum_speech_blocks:
            return ""

        # Estimate the last voiced sample from the trailing VAD silence.
        self.last_speech_end = time.monotonic() - silence_blocks * self.blocksize / rate
        audio = np.concatenate(
            frames,
            axis=0,
        )

        if audio.size == 0:
            return ""

        overall_rms = self._rms(audio)

        if overall_rms < self.minimum_audio_rms:
            return ""

        if rate != 16000:
            audio_16k = resample_poly(
                audio[:, 0],
                16000,
                rate,
            ).astype(np.float32)

        else:
            audio_16k = audio[:, 0].astype(np.float32)

        audio_16k = audio_16k.reshape(
            -1,
            1,
        )

        return self.transcribe(
            audio_16k,
            16000,
        )

    @staticmethod
    def _clean_transcription(
        text,
    ):
        if not text:
            return ""

        lines = [line.strip() for line in str(text).splitlines() if line.strip()]

        if not lines:
            return ""

        recognized = lines[-1].strip()

        recognized = re.sub(
            r"^\[[0-9:.]+\s*-->\s*[0-9:.]+\]\s*",
            "",
            recognized,
        ).strip()

        recognized = re.sub(
            r"^\s*speaker\s*\d*\s*:\s*",
            "",
            recognized,
            flags=re.I,
        ).strip()

        normalized = recognized.strip().lower()

        ignored = {
            "",
            "[blank_audio]",
            "[blank audio]",
            "[silence]",
            "[no speech]",
            "[no speech detected]",
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
            "[noise]",
            "(noise)",
            "[inaudible]",
            "(inaudible)",
            "[applause]",
            "(applause)",
        }

        if normalized in ignored:
            return ""

        bracket_only = normalized.startswith("[") and normalized.endswith("]")

        parentheses_only = normalized.startswith("(") and normalized.endswith(")")

        if bracket_only:
            return ""

        if parentheses_only:
            return ""

        if re.fullmatch(
            r"[\W_]+",
            normalized,
        ):
            return ""

        return recognized

    def transcribe(
        self,
        audio,
        rate,
    ):
        if audio is None or audio.size == 0:
            return ""

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as file:
            path = file.name

        try:
            with wave.open(
                path,
                "wb",
            ) as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(rate)

                pcm = (
                    np.clip(
                        audio[:, 0],
                        -1.0,
                        1.0,
                    )
                    * 32767
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

            started = time.monotonic()

            process = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=120,
            )

            elapsed = time.monotonic() - started

            if process.returncode != 0:
                print(
                    (f"Whisper error: {process.stderr.strip()}"),
                    flush=True,
                )

                return ""

            recognized = self._clean_transcription(process.stdout)

            if recognized:
                print(
                    (f"STT: {recognized} ({elapsed:.2f}s)"),
                    flush=True,
                )

            return recognized

        except Exception as exc:
            print(
                (f"Whisper error: {exc}"),
                flush=True,
            )

            return ""

        finally:
            try:
                os.unlink(path)

            except OSError:
                pass

    def stop_speaking(
        self,
    ):
        """
        Immediately stops current playback.
        """

        self._set_mouth_active(False)

        with self._playback_lock:
            process = self._playback_process

            if process is not None and process.poll() is None:
                try:
                    process.terminate()

                    process.wait(timeout=1.0)

                except Exception:
                    try:
                        process.kill()

                    except Exception:
                        pass

            self._playback_process = None

        self.is_speaking.clear()

    def warmup(self):
        if os.getenv("PIPER_PERSISTENT", "1") != "1" or self._voice_failed:
            return
        with self._synthesis_lock:
            if self._voice_instance is None:
                try:
                    from piper import PiperVoice

                    self._voice_instance = PiperVoice.load(self.voice, use_cuda=False)
                except Exception as exc:
                    self._voice_failed = True
                    print(f"Persistent Piper unavailable; using CLI: {exc}", flush=True)

    def synthesize(self, text, wav_path):
        with self._synthesis_lock:
            self.warmup()
            if self._voice_instance is not None:
                with wave.open(str(wav_path), "wb") as wav_file:
                    self._voice_instance.synthesize_wav(text, wav_file)
                return
            result = subprocess.run(
                [self.piper, "--model", self.voice, "--output_file", str(wav_path)],
                input=text,
                text=True,
                capture_output=True,
                timeout=60,
            )
            if result.returncode:
                raise RuntimeError(f"Piper failed: {result.stderr.strip()}")

    def speak(self, text, set_state, timing=None):
        text = str(text or "").strip()

        if not text:
            return

        self._active_face = self._get_face_from_callback(set_state)

        self.is_speaking.set()

        self._set_mouth_active(False)

        # MILO has the answer, but there is no
        # actual sound yet.
        set_state("thinking")

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as file:
            wav_path = file.name

        try:
            synth_started = time.monotonic()

            self.synthesize(text, wav_path)
            synth_elapsed = time.monotonic() - synth_started

            print(
                (f"TTS ready: {synth_elapsed:.2f}s"),
                flush=True,
            )

            if timing:
                timing.data.setdefault("tts_ms", round(synth_elapsed * 1000, 2))
                if "tts_ready_ms" not in timing.data:
                    timing.mark("tts_ready")
            output_device = self._find_output_device()

            if output_device.startswith("pipewire:"):
                command = [
                    "pw-play",
                    "--target",
                    output_device.split(":", 1)[1],
                    wav_path,
                ]
                playback_name = output_device
            elif output_device in {"default", "pipewire"}:
                command = [
                    "pw-play",
                    wav_path,
                ]
                playback_name = "PipeWire default"
            else:
                command = [
                    "aplay",
                    "-q",
                    "-D",
                    output_device,
                    wav_path,
                ]
                playback_name = output_device

            print(
                (f"Speaker: {playback_name}"),
                flush=True,
            )

            # The mouth starts immediately before the
            # actual playback process is launched.
            set_state("speaking")

            self._set_mouth_active(True)

            with self._playback_lock:
                self._playback_process = subprocess.Popen(
                    command,
                    stdout=(subprocess.DEVNULL),
                    stderr=(subprocess.PIPE),
                    text=True,
                )

                playback = self._playback_process
                if timing and "playback_launch_ms" not in timing.data:
                    timing.mark("playback_launch")

            try:
                _, stderr = playback.communicate(timeout=90)

            except subprocess.TimeoutExpired:
                playback.kill()

                _, stderr = playback.communicate()

            self._set_mouth_active(False)

            if playback.returncode not in {
                0,
                -15,
            }:
                print(
                    (f"Speaker error: {stderr.strip()}"),
                    flush=True,
                )
                self._output_device = None

        except Exception as exc:
            print(
                (f"Speaker error: {exc}"),
                flush=True,
            )

            self._output_device = None

        finally:
            self._set_mouth_active(False)

            with self._playback_lock:
                self._playback_process = None

            self.is_speaking.clear()

            self._active_face = None

            try:
                os.unlink(wav_path)

            except OSError:
                pass
