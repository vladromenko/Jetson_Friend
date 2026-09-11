"""Conservative, inexpensive candidate filter. Memory extraction is not an LLM call."""

import re


def private_turn(text):
    return bool(
        re.search(
            r"\b(?:don't|do not) (?:remember|save|store) (?:this|that)\b", text, re.I
        )
    )


def candidate(text):
    text = str(text).strip()
    if private_turn(text) or len(text) > 1000:
        return None
    # Temporary self-reports never become permanent facts, even on an explicit request.
    if re.search(
        r"\b(?:i am|i'm|i feel) (?:\w+ )?(?:tired|sad|angry|nervous|worried|happy|excited|stressed|frustrated|anxious|lonely)\b",
        text,
        re.I,
    ):
        return None
    explicit = re.match(r"(?:please )?remember(?: that)?\s+(.+)", text, re.I)
    if explicit:
        return {"text": explicit[1], "kind": "fact", "importance": 0.9}
    patterns = [
        (r"\bI (?:prefer|like .* without)\b", "preference"),
        (r"\bI(?:'m| am) (?:studying|learning)\b|\bI study\b", "fact"),
        (
            r"\bmy (?:exam|demonstration|presentation|appointment) (?:is|was moved)\b",
            "goal",
        ),
        (r"\bI (?:want|plan) to (?:finish|complete|learn)\b", "goal"),
    ]
    for pattern, kind in patterns:
        if re.search(pattern, text, re.I) and not text.endswith("?"):
            result = {"text": text, "kind": kind, "importance": 0.75}
            event = re.search(
                r"\bmy (exam|demonstration|presentation|appointment) (?:is|was moved)\b",
                text,
                re.I,
            )
            if event:
                result["fact_key"] = "schedule:" + event[1].lower()
            return result
    return None
