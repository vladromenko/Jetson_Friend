from __future__ import annotations

import re


_OBJECT_REQUEST = re.compile(
    r"\b(?:where\s+(?:did\s+i\s+(?:leave|put)|(?:are|is))|"
    r"have\s+you\s+seen|can\s+you\s+(?:find|locate)|"
    r"(?:please\s+)?find|look\s+for|remember\s+where)\b",
    re.I,
)
_SCENE_REQUEST = (
    "what do you see", "what can you see", "look around", "describe the scene",
    "what is in front", "what's in front", "what am i holding", "what is this",
    "can you see", "do you see", "что ты видишь", "что перед тобой",
)


def visual_intent(text: str) -> tuple[str, str]:
    lowered = text.strip().lower()
    russian_object = re.search(
        r"(?:где(?:\s+я\s+(?:оставил|положил)|\s+(?:лежат|лежит))?|найди|"
        r"ты\s+(?:видишь|видел))\s+(?:(?:мой|мои|моя|моё)\s+)?([а-яё][а-яё-]{1,30})",
        lowered,
    )
    if _OBJECT_REQUEST.search(lowered) or russian_object:
        match = re.search(r"\b(?:my|the|our)\s+([a-z][a-z0-9' -]{0,48})", lowered)
        if match:
            label = re.split(r"\b(?:that|which|when|please|right now|today|last time|on|in|under|near)\b", match.group(1))[0]
            return "object", label.strip(" -?.!,")
        return "object", russian_object.group(1) if russian_object else ""
    if any(phrase in lowered for phrase in _SCENE_REQUEST):
        return "scene", ""
    return "chat", ""
