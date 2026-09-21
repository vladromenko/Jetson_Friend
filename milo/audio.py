from __future__ import annotations

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
from pathlib import Path


class VoiceIO:
    def __init__(self, cfg, on_speaking=None):
        self.cfg = cfg
        self.on_speaking = on_speaking or (lambda _: None)
        self.is_speaking = threading.Event()
        self._speak_lock = threading.Lock()
        self.input_device = None
        self.output_device = None
        self._validated_capture = None
        self._init_sounddevice()
        self.output_device = self._find_output_device()
        input_name = self._input_name() if self.sd is not None else "arecord fallback"
        print(f"[AUDIO] microphone={input_name}", flush=True)
        print(f"[AUDIO] speaker={self.output_device}", flush=True)

    def _init_sounddevice(self):
        try:
            import numpy as np
            import sounddevice as sd
            from scipy.signal import resample_poly
            self.np = np
            self.sd = sd
            self.resample_poly = resample_poly
            self.input_device = self._find_input_device()
        except Exception as exc:
            self.np = None
            self.sd = None
            self.resample_poly = None
            print(f"[AUDIO] sounddevice VAD unavailable ({exc}); using ALSA fallback", flush=True)

    def _find_input_device(self):
        devices = self.sd.query_devices()
        wanted = self.cfg.audio_input_hint.strip().lower()
        if wanted:
            for index, device in enumerate(devices):
                name = str(device.get("name", ""))
                if int(device.get("max_input_channels", 0)) > 0 and wanted in name.lower():
                    return index
        for index, device in enumerate(devices):
            name = str(device.get("name", "")).lower()
            if int(device.get("max_input_channels", 0)) > 0 and any(token in name for token in ("usb", "c920", "uac", "webcam")):
                return index
        try:
            default_input = int(self.sd.default.device[0])
            if default_input >= 0 and int(devices[default_input].get("max_input_channels", 0)) > 0:
                return default_input
        except Exception:
            pass
        return None

    def _input_name(self):
        if self.sd is None or self.input_device is None:
            return "none"
        try:
            return str(self.sd.query_devices(self.input_device, "input").get("name", self.input_device))
        except Exception:
            return str(self.input_device)

    def _find_output_device(self):
        configured = self.cfg.audio_output_hint.strip()
        if configured.startswith(("hw:", "plughw:", "default", "pipewire:")):
            return configured
        if shutil.which("pw-play") and shutil.which("pactl"):
            try:
                sinks = subprocess.run(["pactl", "list", "short", "sinks"], text=True, capture_output=True, timeout=5)
                if sinks.returncode == 0 and configured:
                    for line in sinks.stdout.splitlines():
                        fields = line.split()
                        if len(fields) >= 2 and configured.lower() in fields[1].lower():
                            return f"pipewire:{fields[1]}"
            except Exception:
                pass
        try:
            process = subprocess.run(["aplay", "-l"], text=True, capture_output=True, timeout=5)
            selected = None
            for line in process.stdout.splitlines():
                lower = line.lower()
                wanted = configured.lower()
                if selected is None and "card " in lower and wanted and wanted in lower:
                    match = re.search(r"card\s+(\d+):.*device\s+(\d+):", line, flags=re.I)
                    if match:
                        selected = f"plughw:{match.group(1)},{match.group(2)}"
            if selected is None:
                for line in process.stdout.splitlines():
                    lower = line.lower()
                    if "card " in lower and ("uacdemo" in lower or "usb audio" in lower):
                        match = re.search(r"card\s+(\d+):.*device\s+(\d+):", line, flags=re.I)
                        if match:
                            selected = f"plughw:{match.group(1)},{match.group(2)}"
                            break
            return selected or "default"
        except Exception:
            return "default"

    @staticmethod
    def _clean_transcription(text: str) -> str:
        if not text:
            return ""
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        if not lines:
            return ""
        recognized = lines[-1]
        recognized = re.sub(r"^\[[0-9:.]+\s*-->\s*[0-9:.]+\]\s*", "", recognized).strip()
        recognized = re.sub(r"^\s*speaker\s*\d*\s*:\s*", "", recognized, flags=re.I).strip()
        normalized = recognized.lower().strip()
        ignored = {
            "", "[blank_audio]", "[blank audio]", "[silence]", "[no speech]", "[no speech detected]",
            "[music]", "(music)", "(sighs)", "[sighs]", "(sigh)", "[sigh]", "(breathing)", "[breathing]",
            "(background noise)", "[background noise]", "[noise]", "(noise)", "[inaudible]", "(inaudible)",
            "[applause]", "(applause)", "(wind blowing)", "[wind blowing]", "(wind)", "[wind]",
        }
        if normalized in ignored:
            return ""
        if (normalized.startswith("[") and normalized.endswith("]")) or (normalized.startswith("(") and normalized.endswith(")")) or (normalized.startswith("*") and normalized.endswith("*")):
            return ""
        if re.fullmatch(r"[\W_]+", normalized):
            return ""
        return recognized

    @staticmethod
    def _rms(chunk) -> float:
        import numpy as np
        if chunk is None or chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32))))

    def _capture_rate(self, device, requested):
        cached = self._validated_capture
        if cached and cached[:2] == (device, requested):
            return cached[2]
        native = int(self.sd.query_devices(device, "input")["default_samplerate"])
        for rate in dict.fromkeys((requested, 16000, native)):
            try:
                self.sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=rate)
                self._validated_capture = (device, requested, rate)
                return rate
            except Exception:
                pass
        raise RuntimeError("Microphone has no supported mono capture format")

    def _listen_vad(self, stop_event=None) -> str:
        if self.input_device is None:
            return ""
        rate = self._capture_rate(self.input_device, self.cfg.mic_sample_rate)
        blocksize = self.cfg.mic_block_size
        audio_queue = queue.Queue(maxsize=64)
        frames = []
        pre_roll_blocks = max(1, int(self.cfg.vad_pre_roll * rate / blocksize))
        pre_roll = deque(maxlen=pre_roll_blocks)
        silence_needed = max(2, int(self.cfg.vad_end_silence * rate / blocksize))
        minimum_speech_blocks = max(1, int(self.cfg.vad_min_speech * rate / blocksize))
        speaking = False
        speech_blocks = 0
        silence_blocks = 0
        finished = False
        started_at = time.monotonic()

        def callback(indata, frames_count, time_info, status):
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

        with self.sd.InputStream(device=self.input_device, channels=1, samplerate=rate, blocksize=blocksize, dtype="float32", callback=callback):
            while not finished:
                if stop_event is not None and stop_event.is_set():
                    return ""
                if self.is_speaking.is_set():
                    return ""
                if time.monotonic() - started_at >= self.cfg.mic_max_seconds:
                    finished = True
                else:
                    try:
                        chunk = audio_queue.get(timeout=0.20)
                    except queue.Empty:
                        chunk = None
                    if chunk is not None:
                        rms = self._rms(chunk)
                        if not speaking:
                            pre_roll.append(chunk)
                            if rms >= self.cfg.vad_speech_threshold:
                                speaking = True
                                frames.extend(list(pre_roll))
                                pre_roll.clear()
                                speech_blocks = 1
                                silence_blocks = 0
                        else:
                            frames.append(chunk)
                            if rms >= self.cfg.vad_silence_threshold:
                                speech_blocks += 1
                                silence_blocks = 0
                            else:
                                silence_blocks += 1
                            if silence_blocks >= silence_needed and speech_blocks >= minimum_speech_blocks:
                                finished = True

        if not speaking or not frames or speech_blocks < minimum_speech_blocks:
            return ""
        audio = self.np.concatenate(frames, axis=0)
        if audio.size == 0 or self._rms(audio) < self.cfg.whisper_min_audio_rms:
            return ""
        if rate != 16000:
            samples = self.resample_poly(audio[:, 0], 16000, rate).astype(self.np.float32)
        else:
            samples = audio[:, 0].astype(self.np.float32)
        return self._transcribe_float(samples, 16000)

    def _transcribe_float(self, samples, rate: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = Path(handle.name)
        try:
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(rate)
                pcm = (self.np.clip(samples, -1.0, 1.0) * 32767).astype(self.np.int16)
                wav_file.writeframes(pcm.tobytes())
            command = [
                str(self.cfg.whisper_bin), "-m", str(self.cfg.whisper_model), "-f", str(path),
                "-l", self.cfg.whisper_language, "-nt", "-np",
            ]
            process = subprocess.run(command, text=True, capture_output=True, timeout=120)
            if process.returncode != 0:
                print(f"[STT] whisper error: {process.stderr.strip()}", flush=True)
                return ""
            recognized = self._clean_transcription(process.stdout)
            if recognized:
                print(f"[STT] {recognized}", flush=True)
            return recognized
        finally:
            try:
                path.unlink()
            except OSError:
                pass

    def _listen_arecord(self) -> str:
        with tempfile.TemporaryDirectory(prefix="milo_audio_") as tmp:
            wav = Path(tmp) / "input.wav"
            command = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "4", str(wav)]
            try:
                subprocess.run(command, check=True, timeout=8)
                with wave.open(str(wav), "rb") as wf:
                    frames = wf.readframes(wf.getnframes())
                import array
                samples = array.array("h")
                samples.frombytes(frames)
                if not samples:
                    return ""
                rms = (sum(int(x) * int(x) for x in samples) / len(samples)) ** 0.5
                if rms < 250:
                    return ""
                process = subprocess.run([
                    str(self.cfg.whisper_bin), "-m", str(self.cfg.whisper_model), "-f", str(wav),
                    "-l", self.cfg.whisper_language, "-nt", "-np",
                ], text=True, capture_output=True, timeout=120)
                return self._clean_transcription(process.stdout)
            except Exception as exc:
                print(f"[AUDIO] ALSA fallback failed: {exc}", flush=True)
                return ""

    def listen(self, stop_event=None) -> str:
        if self.is_speaking.is_set():
            return ""
        if self.sd is not None:
            try:
                return self._listen_vad(stop_event)
            except Exception as exc:
                print(f"[AUDIO] VAD capture failed: {exc}", flush=True)
                self.input_device = self._find_input_device()
                return ""
        return self._listen_arecord()

    def speak(self, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        with self._speak_lock:
            self._speak(text)

    def _speak(self, text: str) -> None:
        with tempfile.TemporaryDirectory(prefix="milo_tts_") as tmp:
            wav = Path(tmp) / "reply.wav"
            playback_started = False
            try:
                process = subprocess.run(
                    [str(self.cfg.piper_bin), "--model", str(self.cfg.piper_voice), "--output_file", str(wav)],
                    input=text, text=True, capture_output=True, timeout=60,
                )
                if process.returncode != 0:
                    raise RuntimeError(process.stderr.strip() or "Piper failed")
                if self.output_device.startswith("pipewire:") and shutil.which("pw-play"):
                    command = ["pw-play", "--target", self.output_device.split(":", 1)[1], str(wav)]
                else:
                    command = ["aplay", "-q"]
                    if self.output_device and self.output_device != "default":
                        command += ["-D", self.output_device]
                    command.append(str(wav))
                self.is_speaking.set()
                self.on_speaking(True)
                playback_started = True
                subprocess.run(command, check=False, timeout=60)
            except Exception as exc:
                print(f"[TTS] failed: {exc}", flush=True)
            finally:
                if playback_started:
                    self.on_speaking(False)
                    self.is_speaking.clear()
