import re


def wants_visual(text):
    return bool(
        re.search(
            r"\b(?:look (?:at|around|through)|what (?:can|do) you see|"
            r"describe (?:me|the room|this|that|your surroundings|what you see)|"
            r"what (?:am i|is he|is she) (?:wearing|holding)|"
            r"who(?: is|'s) (?:here|in front of you)|what(?: is|'s) this|"
            r"(?:check|use|show me) (?:the |your )?camera)\b",
            str(text),
            re.I,
        )
    )
