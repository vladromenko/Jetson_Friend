from __future__ import annotations

import base64
import json
import re
import urllib.request


class LocalModel:
    def __init__(self, base_url: str, model_name: str, max_tokens: int = 180, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.history: list[dict[str, str]] = []

    def _post(self, payload: dict, timeout: int | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _message_text(message: dict) -> str:
        raw = message.get("content")
        if isinstance(raw, list):
            raw = "".join(str(item.get("text", "")) for item in raw if isinstance(item, dict))
        text = str(raw or "").strip()
        if not text:
            text = str(message.get("reasoning_content") or "").strip()
        if "<think>" in text and "</think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return text

    def _extract(self, response: dict) -> str:
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return self._message_text(message)

    def _base_payload(self, messages):
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def remember_exchange(self, user_text: str, answer: str) -> None:
        self.history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}])
        self.history = self.history[-12:]

    @staticmethod
    def _mood_guidance(text: str) -> str:
        lowered = text.lower()
        if (
            re.search(r"\b(?:i(?:'m| am)(?: feeling)?|i feel|feeling)\s+(?:really |very )?(?:sad|lonely|down|upset|scared|anxious|depressed|hurt)\b", lowered)
            or re.search(r"\b(?:i'?m\s+crying|i\s+am\s+crying|i\s+cry|crying)\b", lowered)
            or re.search(r"(?:мне\s+(?:грустно|одиноко|страшно|тревожно|плохо)|я\s+плачу|я\s+расстроен|я\s+расстроена)", lowered)
        ):
            return (
                "The user is sharing a vulnerable or painful feeling. Generate a genuinely supportive reply in clear spoken English. "
                "First acknowledge the feeling in concrete words, then reassure them warmly that this moment can pass and they can handle the next small step. "
                "You may ask one gentle question. Do not sound like a generic assistant, do not minimize, and do not use a fixed template."
            )
        if re.search(r"\b(?:i(?:'m| am)(?: feeling)?|i feel|feeling)\s+(?:really |very )?(?:happy|excited|proud|glad)\b", lowered) or re.search(r"мне\s+(?:радостно|весело|хорошо)", lowered):
            return "The user is sharing good news. Respond in clear spoken English with warm, specific enthusiasm and invite them to tell you more."
        return ""

    def chat(self, user_text: str) -> str:
        system = (
            "You are MILO, a warm, attentive voice assistant beside the user. Reply in clear spoken English, "
            "even if the user mixes in Russian. "
            "usually in one to three short spoken sentences. Notice emotional meaning: when the user feels low, "
            "acknowledge the specific feeling before offering support and reassurance; when they are happy, share their excitement. "
            "Ask at most one natural follow-up question. Avoid generic lines such as 'How can I help you today?' "
            "Do not claim to see, remember, or know a physical location unless camera evidence or a recorded sighting exists. "
            "Do not claim to have human feelings. Never return an empty answer."
        )
        messages = [{"role": "system", "content": system}]
        mood = self._mood_guidance(user_text)
        if mood:
            messages.append({"role": "system", "content": mood})
        messages += self.history[-8:] + [{"role": "user", "content": user_text}]
        response = self._post(self._base_payload(messages))
        answer = self._extract(response)
        generic = r"\b(?:how can i help|is there anything i can|i don't understand|please clarify|чем могу помочь|как я могу помочь)\b"
        if mood and re.search(generic, answer.lower()):
            repair = self._base_payload(messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Rewrite your previous reply. Make it emotionally supportive, concrete, and natural. Reassure me gently. Do not use a generic assistant phrase."},
            ])
            try:
                improved = self._extract(self._post(repair))
            except Exception as exc:
                print(f"[LLM] emotional retry failed: {exc}", flush=True)
                improved = ""
            if improved:
                answer = improved
        if not answer:
            print("[LLM] empty response: " + json.dumps(response, ensure_ascii=False)[:1800], flush=True)
            answer = "I heard you, but my local model returned an empty response. Please say that again."
        self.remember_exchange(user_text, answer)
        return answer


    def proactive_reply(self, instruction: str) -> str:
        system = (
            "You are MILO, a small embodied desktop robot. Reply only in clear spoken English. "
            "Be brief, playful, and warm. You may be a little cheeky, but never mean. One short spoken sentence."
        )
        response = self._post(self._base_payload([
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]))
        return self._extract(response).strip()

    def describe(self, user_text: str, jpeg: bytes) -> str:
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        payload = self._base_payload([
            {"role": "system", "content": "Use the image to answer the user's question. Describe only visible evidence; do not guess unseen objects or past locations. Reply naturally in clear spoken English."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ])
        payload["temperature"] = 0.2
        response = self._post(payload, timeout=max(self.timeout, 120))
        answer = self._extract(response)
        if not answer:
            print("[VLM] empty response: " + json.dumps(response, ensure_ascii=False)[:1800], flush=True)
            return "I can see the camera frame, but my visual model returned an empty description."
        self.remember_exchange(user_text, answer)
        return answer

    def locate_object(self, user_text: str, label: str, jpeg: bytes) -> dict:
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        prompt = (
            "Inspect this camera image for the requested item. Request: " + user_text +
            ". Item hint: " + (label or "infer the item from the request") +
            '. Return only JSON: {"object":"short item name","visible":true or false,'
            '"location":"brief location relative to visible landmarks, or empty string"}. '
            "Set visible=true only if you can clearly identify the requested item. "
            "Write location in English. Do not guess where it was left."
        )
        payload = self._base_payload([{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }])
        payload["temperature"] = 0.0
        response = self._post(payload, timeout=max(self.timeout, 120))
        raw = self._extract(response)
        try:
            start = raw.index("{")
            result, _ = json.JSONDecoder().raw_decode(raw[start:])
        except (ValueError, TypeError):
            print(f"[VLM] could not parse object location: {raw[:300]}", flush=True)
            return {"object": label, "visible": False, "location": "", "reliable": False}
        if not isinstance(result, dict):
            return {"object": label, "visible": False, "location": "", "reliable": False}
        visible = result.get("visible") is True
        location = str(result.get("location") or "").strip() if visible else ""
        model_object = str(result.get("object") or "").strip()
        if label and model_object and label.lower() not in model_object.lower() and model_object.lower() not in label.lower():
            visible = False
            location = ""
        if re.search(r"\b(?:not visible|unknown|not sure|can't tell|cannot tell)\b", location.lower()):
            visible = False
            location = ""
        return {"object": label or model_object or "item", "visible": visible and bool(location), "location": location, "reliable": True}

    def observe_person(self, jpeg: bytes) -> dict:
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        prompt = (
            'Look only at visible evidence in this camera image. Return only JSON: '
            '{"person_visible":true or false,"phone_visible":true or false,'
            '"idle":true or false,"confidence":0.0 to 1.0}. '
            "phone_visible=true only if a phone is clearly visible in the person's hands or near the person. "
            "idle=true only if the person appears present and not doing an obvious task. Do not guess."
        )
        payload = self._base_payload([{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }])
        payload["temperature"] = 0.0
        payload["max_tokens"] = 80
        response = self._post(payload, timeout=max(self.timeout, 120))
        raw = self._extract(response)
        try:
            start = raw.index("{")
            result, _ = json.JSONDecoder().raw_decode(raw[start:])
        except (ValueError, TypeError):
            print(f"[VLM] could not parse person observation: {raw[:300]}", flush=True)
            return {"person_visible": False, "phone_visible": False, "idle": False, "confidence": 0.0}
        if not isinstance(result, dict):
            return {"person_visible": False, "phone_visible": False, "idle": False, "confidence": 0.0}
        return {
            "person_visible": result.get("person_visible") is True,
            "phone_visible": result.get("phone_visible") is True,
            "idle": result.get("idle") is True,
            "confidence": float(result.get("confidence") or 0.0),
        }

    def health_check(self) -> str:
        payload = self._base_payload([{"role": "user", "content": "Reply with the single word READY."}])
        payload["temperature"] = 0.0
        payload["max_tokens"] = 12
        response = self._post(payload, timeout=self.timeout)
        answer = self._extract(response)
        if not answer:
            raise RuntimeError("local model returned an empty health-check response")
        return answer
