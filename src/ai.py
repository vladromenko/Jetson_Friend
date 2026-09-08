import json
import os
import re
import subprocess

SYSTEM_PROMPT = """You are Hugh, a small local AI robot companion running on an NVIDIA Jetson Orin Nano.
You interact through a camera, microphone, speaker and animated face.
You are calm, intelligent, curious and mildly sarcastic.
Speak naturally and concisely in English.
You may use visual sensor data supplied by the application, but never claim to see an object that was not detected.
Your name is Hugh.
Return only JSON like {"text":"...","emotion":"neutral"}."""

EMOTIONS = {"neutral", "happy", "thinking", "confused", "curious", "surprised", "concerned", "sad"}


class HughAI:
    def __init__(self):
        self.bin = os.getenv("LLAMA_BIN", "/app/deps/llama.cpp/build/bin/llama-cli")
        self.model = os.getenv("LLM_MODEL", "/app/models/llm/Qwen3-4B-Q4_K_M.gguf")
        self.ctx = os.getenv("LLAMA_CTX", "4096")
        self.gpu_layers = os.getenv("LLAMA_GPU_LAYERS", "99")
        self.history = []
        self.last_gpu_info = "unknown"

    def ask(self, user_text, scene=None):
        scene_text = json.dumps(scene or {}, separators=(",", ":"))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-10:])
        messages.append({"role": "user", "content": f"Scene state: {scene_text}\nUser: {user_text}"})
        prompt = self._chatml(messages)
        cmd = [
            self.bin, "-m", self.model, "-c", self.ctx, "-ngl", self.gpu_layers,
            "-n", "160", "--temp", "0.6", "--no-display-prompt", "-p", prompt,
        ]
        try:
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
            self.last_gpu_info = "CUDA/offload requested" if "-ngl" in cmd else "CPU"
            raw = p.stdout.strip()
            if p.returncode:
                return {"text": "My language model tripped over a cable. Metaphorically, sadly.", "emotion": "confused"}
            reply = self._parse(raw)
        except Exception as exc:
            reply = {"text": f"I cannot reach my local brain: {exc}", "emotion": "confused"}
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply["text"]}]
        self.history = self.history[-10:]
        return reply

    def _chatml(self, messages):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"

    def _parse(self, raw):
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            data = json.loads(match.group(0) if match else raw)
            text = str(data.get("text", "")).strip() or "I have nothing useful to add. A rare treat."
            emotion = str(data.get("emotion", "neutral")).lower()
        except Exception:
            text, emotion = raw.strip(), "neutral"
        if emotion not in EMOTIONS:
            emotion = "neutral"
        return {"text": text[:600], "emotion": emotion}
