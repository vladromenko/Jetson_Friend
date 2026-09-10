import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request


SYSTEM_PROMPT = """
/no_think
You are Hugh, a small local AI companion.

Personality:
- warm, calm, emotionally supportive and natural,
- answer the user's CURRENT message first,
- usually reply in one or two short spoken sentences,
- when the user explicitly says they feel sad, nervous, lonely or upset, respond with genuine warmth,
- do not repeatedly bring up an old emotion unless the user brings it up again,
- occasionally be playful, but never forced or overly cheerful,
- never sound like customer support.

Memory:
- memories are background context only,
- use a memory only when it clearly helps answer the current message,
- never let an unrelated memory override the current question,
- do not repeatedly save the same fact or emotional state,
- save only durable preferences, goals, important facts or meaningful personal events,
- temporary moods normally should not become permanent memories.

Vision:
- use Scene state only as detector information,
- never invent visual details.

Return ONLY compact JSON:
{"text":"reply","emotion":"neutral","memory":{"save":false,"kind":"fact","text":"","emotion":"neutral","importance":0.5}}

Allowed emotions:
neutral, happy, thinking, confused, curious, surprised, concerned, sad.
""".strip()


EMOTIONS = {
    "neutral",
    "happy",
    "thinking",
    "confused",
    "curious",
    "surprised",
    "concerned",
    "sad",
}


class HughAI:
    def __init__(self):
        root = os.getenv(
            "JETSON_FRIEND_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        self.model = os.getenv(
            "LLM_MODEL",
            os.path.join(
                root,
                "models",
                "llm",
                "Qwen3-4B-Q4_K_M.gguf",
            ),
        )

        self.server_bin = os.getenv(
            "LLAMA_SERVER_BIN",
            os.path.join(
                root,
                "deps",
                "llama.cpp",
                "build",
                "bin",
                "llama-server",
            ),
        )

        self.server_host = os.getenv(
            "LLAMA_SERVER_HOST",
            "127.0.0.1",
        )

        self.server_port = int(
            os.getenv(
                "LLAMA_SERVER_PORT",
                "8081",
            )
        )

        self.server_url = os.getenv(
            "LLAMA_SERVER_URL",
            (
                f"http://{self.server_host}:"
                f"{self.server_port}/v1/chat/completions"
            ),
        )

        self.context_size = int(
            os.getenv(
                "LLAMA_CTX",
                "3072",
            )
        )

        self.gpu_layers = os.getenv(
            "LLAMA_GPU_LAYERS",
            "99",
        )

        self.history = []
        self.server_process = None

        self.ensure_server()

    def _server_alive(self):
        try:
            request = urllib.request.Request(
                f"http://{self.server_host}:{self.server_port}/health",
                method="GET",
            )

            with urllib.request.urlopen(
                request,
                timeout=2,
            ) as response:
                return response.status == 200

        except Exception:
            return False

    def ensure_server(self):
        if self._server_alive():
            print(
                "LLM server already running.",
                flush=True,
            )
            return

        if not os.path.isfile(self.server_bin):
            raise RuntimeError(
                f"llama-server not found: {self.server_bin}"
            )

        if not os.path.isfile(self.model):
            raise RuntimeError(
                f"LLM model not found: {self.model}"
            )

        gpu_layers = self.gpu_layers

        if not str(gpu_layers).isdigit():
            gpu_layers = "99"

        command = [
            self.server_bin,
            "-m",
            self.model,
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
            "-c",
            str(self.context_size),
            "-ngl",
            str(gpu_layers),
            "--flash-attn",
            "on",
            "--parallel",
            "1",
        ]

        print(
            "Starting local Qwen server...",
            flush=True,
        )

        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        deadline = time.monotonic() + 45.0

        while (
            time.monotonic() < deadline
            and not self._server_alive()
        ):
            if self.server_process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited during startup"
                )

            time.sleep(0.5)

        if not self._server_alive():
            raise RuntimeError(
                "llama-server did not become ready"
            )

        print(
            "Local Qwen server ready.",
            flush=True,
        )

    def ask(
        self,
        user_text,
        scene=None,
        person_name=None,
        memory_context=None,
    ):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        messages.extend(
            self.history[-4:]
        )

        name = (
            person_name
            if person_name
            else "unknown person"
        )

        lower_text = user_text.lower()

        vision_words = (
            "see", "look", "camera", "wearing", "holding",
            "describe me", "who is here", "who's here",
            "what is here", "around me", "in front of you"
        )

        wants_vision = any(
            word in lower_text
            for word in vision_words
        )

        parts = [
            f"Person: {name}",
        ]

        if wants_vision and scene:
            scene_text = json.dumps(
                scene,
                separators=(",", ":"),
            )
            parts.append(
                f"Scene: {scene_text}"
            )

        if (
            memory_context
            and memory_context != "No relevant long-term memories."
        ):
            parts.append(
                "Relevant memory:\n" + memory_context
            )

        parts.append(
            "Current message: " + user_text
        )

        prompt = "\n".join(parts)

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "messages": messages,
            "temperature": 0.55,
            "top_p": 0.85,
            "max_tokens": 80,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }

        request = urllib.request.Request(
            self.server_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            started = time.monotonic()

            with urllib.request.urlopen(
                request,
                timeout=90,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

            raw = data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]

            reply = self._parse(
                raw
            )

            elapsed = (
                time.monotonic()
                - started
            )

            print(
                f"LLM response time: {elapsed:.2f}s",
                flush=True,
            )

        except urllib.error.URLError as exc:
            print(
                f"LLM server error: {exc}",
                flush=True,
            )

            reply = self._fallback(
                "My local brain isn't responding right now."
            )

        except Exception as exc:
            print(
                f"LLM error: {exc}",
                flush=True,
            )

            reply = self._fallback(
                "Something went wrong in my local brain."
            )

        self.history.extend(
            [
                {
                    "role": "user",
                    "content": user_text,
                },
                {
                    "role": "assistant",
                    "content": reply["text"],
                },
            ]
        )

        self.history = self.history[-4:]

        return reply

    def _fallback(self, text):
        return {
            "text": text,
            "emotion": "concerned",
            "memory": {
                "save": False,
                "kind": "fact",
                "text": "",
                "emotion": "neutral",
                "importance": 0.5,
            },
        }

    def _parse(self, raw):
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            raw,
            flags=re.S,
        ).strip()

        candidates = re.findall(
            r"\{.*\}",
            cleaned,
            flags=re.S,
        )

        data = None

        for candidate in reversed(
            candidates
        ):
            try:
                parsed = json.loads(
                    candidate
                )
            except Exception:
                parsed = None

            if (
                isinstance(parsed, dict)
                and "text" in parsed
            ):
                data = parsed
                break

        if data is None:
            return {
                "text": cleaned[:600] or "Okay.",
                "emotion": "neutral",
                "memory": {
                    "save": False,
                    "kind": "fact",
                    "text": "",
                    "emotion": "neutral",
                    "importance": 0.5,
                },
            }

        emotion = str(
            data.get(
                "emotion",
                "neutral",
            )
        ).lower()

        if emotion not in EMOTIONS:
            emotion = "neutral"

        memory = data.get(
            "memory",
            {},
        )

        if not isinstance(
            memory,
            dict,
        ):
            memory = {}

        memory_emotion = str(
            memory.get(
                "emotion",
                "neutral",
            )
        ).lower()

        if memory_emotion not in EMOTIONS:
            memory_emotion = "neutral"

        try:
            importance = float(
                memory.get(
                    "importance",
                    0.5,
                )
            )
        except Exception:
            importance = 0.5

        importance = max(
            0.0,
            min(
                1.0,
                importance,
            ),
        )

        return {
            "text": str(
                data.get(
                    "text",
                    "Okay.",
                )
            )[:600],
            "emotion": emotion,
            "memory": {
                "save": bool(
                    memory.get(
                        "save",
                        False,
                    )
                ),
                "kind": str(
                    memory.get(
                        "kind",
                        "fact",
                    )
                )[:30],
                "text": str(
                    memory.get(
                        "text",
                        "",
                    )
                )[:1000],
                "emotion": memory_emotion,
                "importance": importance,
            },
        }

    def close(self):
        if (
            self.server_process is not None
            and self.server_process.poll() is None
        ):
            self.server_process.terminate()

            try:
                self.server_process.wait(
                    timeout=5,
                )
            except subprocess.TimeoutExpired:
                self.server_process.kill()
