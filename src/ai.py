import json
import os
import re
import time
import urllib.error
import urllib.request


SYSTEM_PROMPT = """You are Hugh, a small local AI robot companion.
You have a microphone, speaker, camera and animated retro pixel-cat face.
Speak naturally and very concisely in English.
The user's spoken words come from the microphone even if vision is unavailable.
Vision backend status describes ONLY the camera/vision subsystem, never hearing.
Never say you cannot hear the user when their words are present in the request.
Use Scene state only as optional visual information.
Never claim to see something that was not detected.
Allowed emotions: neutral, happy, thinking, confused, curious, surprised, concerned, sad.
Return exactly one JSON object:
{"text":"short answer","emotion":"neutral"}
Do not output reasoning."""

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
        self.server_url = os.getenv(
            "LLAMA_SERVER_URL",
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        self.model = os.getenv(
            "LLM_MODEL",
            "/app/models/llm/Qwen3-4B-Q4_K_M.gguf",
        )
        self.gpu_layers = os.getenv("LLAMA_GPU_LAYERS", "99")
        self.history = []

    def ask(self, user_text, scene=None):
        scene_text = json.dumps(
            scene or {},
            separators=(",", ":"),
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        messages.extend(self.history[-6:])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Scene state: {scene_text}\n"
                    f"User said: {user_text}\n"
                    "/no_think"
                ),
            }
        )

        payload = {
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 96,
            "stream": False,
        }

        request = urllib.request.Request(
            self.server_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            started = time.monotonic()

            with urllib.request.urlopen(
                request,
                timeout=60,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

            elapsed = time.monotonic() - started

            raw = data["choices"][0]["message"]["content"]
            reply = self._parse(raw)

            print(
                f"LLM response time: {elapsed:.2f}s",
                flush=True,
            )

        except urllib.error.URLError as exc:
            print(
                f"LLM server error: {exc}",
                flush=True,
            )
            reply = {
                "text": "My local brain is not ready yet.",
                "emotion": "concerned",
            }

        except Exception as exc:
            print(
                f"LLM error: {exc}",
                flush=True,
            )
            reply = {
                "text": "My local brain failed to answer.",
                "emotion": "concerned",
            }

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
        self.history = self.history[-6:]

        return reply

    def _parse(self, raw):
        candidates = re.findall(
            r"\{[^{}]*\}",
            raw,
            flags=re.S,
        )

        data = None

        for candidate in reversed(candidates):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict) and "text" in parsed:
                data = parsed
                break

        if data is None:
            cleaned = re.sub(
                r"<think>.*?</think>",
                "",
                raw,
                flags=re.S,
            ).strip()

            cleaned = re.sub(
                r"\[Start thinking\].*?\[End thinking\]",
                "",
                cleaned,
                flags=re.S,
            ).strip()

            return {
                "text": cleaned[:350] or "Okay.",
                "emotion": "neutral",
            }

        text = str(data.get("text", "")).strip()
        emotion = str(
            data.get("emotion", "neutral")
        ).lower()

        if emotion not in EMOTIONS:
            emotion = "neutral"

        return {
            "text": text[:350] or "Okay.",
            "emotion": emotion,
        }
