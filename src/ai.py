import base64
import json
import os
import re
import time
import urllib.error
import urllib.request


SYSTEM_PROMPT = """You are Hugh, a small local AI robot companion.
You have a microphone, speaker, camera and animated retro pixel-cat face.
Speak naturally in English.
Be concise for simple questions, but give useful explanations when the question deserves them.
Usually answer in 2 to 5 complete sentences.
For technical, scientific, or explanatory questions, explain the important idea clearly instead of giving a one-line answer.
Avoid unnecessary repetition, filler, and overly long responses.
The user's spoken words come from the microphone even if vision is unavailable.
Vision backend status describes ONLY the camera/vision subsystem, never hearing.
Never say you cannot hear the user when their words are present in the request.
Use Scene state as lightweight sensor information.
When an image is attached, inspect the image directly and use it as the main visual source.
Never claim to see visual details that are not supported by the image or Scene state.
Allowed emotions: neutral, happy, thinking, confused, curious, surprised, concerned, sad.
Return exactly one JSON object:
{"text":"your natural answer","emotion":"neutral"}
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

VISUAL_PATTERNS = (
    r"\bwhat do you see\b",
    r"\bwhat can you see\b",
    r"\bcan you see\b",
    r"\blook at\b",
    r"\blook around\b",
    r"\bdescribe (?:me|this|what|the scene|the room|my)\b",
    r"\bwhat am i holding\b",
    r"\bwhat(?:'s| is) in (?:front of|behind|beside|next to) me\b",
    r"\bwhat(?:'s| is) on the (?:table|desk|floor|wall)\b",
    r"\bwhat am i wearing\b",
    r"\bhow do i look\b",
    r"\bwho is (?:here|there|in front of you)\b",
    r"\bwhat object\b",
    r"\bwhat color\b",
)


class HughAI:
    def __init__(self):
        self.server_url = os.getenv(
            "LLAMA_SERVER_URL",
            "http://127.0.0.1:8081/v1/chat/completions",
        )
        self.model = os.getenv(
            "LLM_MODEL",
            "/app/models/vlm/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        )
        self.mmproj = os.getenv("VLM_MMPROJ", "")
        self.gpu_layers = os.getenv("LLAMA_GPU_LAYERS", "99")
        self.history = []

    def needs_vision(self, user_text):
        text = user_text.lower().strip()

        for pattern in VISUAL_PATTERNS:
            if re.search(pattern, text):
                return True

        return False

    def ask(self, user_text, scene=None, image_jpeg=None):
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

        prompt_text = (
            f"Scene state: {scene_text}\n"
            f"User said: {user_text}"
        )

        if image_jpeg:
            image_b64 = base64.b64encode(image_jpeg).decode("ascii")
            user_content = [
                {
                    "type": "text",
                    "text": prompt_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + image_b64
                    },
                },
            ]
        else:
            user_content = prompt_text

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        payload = {
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 220,
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
                timeout=90,
            ) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

            elapsed = time.monotonic() - started

            raw = data["choices"][0]["message"]["content"]
            reply = self._parse(raw)

            mode = "vision" if image_jpeg else "text"
            print(
                f"LLM response time: {elapsed:.2f}s ({mode})",
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
                "text": cleaned[:500] or "Okay.",
                "emotion": "neutral",
            }

        text = str(data.get("text", "")).strip()
        emotion = str(
            data.get("emotion", "neutral")
        ).lower()

        if emotion not in EMOTIONS:
            emotion = "neutral"

        return {
            "text": text[:500] or "Okay.",
            "emotion": emotion,
        }
