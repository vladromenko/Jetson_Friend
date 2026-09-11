"""SSE transport and conservative phrase splitting, independent of audio devices."""

import json
import re


def deltas(response):
    """Yield decoded OpenAI-compatible SSE events; reject truncated streams."""
    data_lines = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif not line and data_lines:
            payload = "\n".join(data_lines)
            data_lines = []
            if payload == "[DONE]":
                return
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError("LLM stream returned an error")
            yield event
    raise RuntimeError("LLM stream ended without [DONE]")


class Phrases:
    def __init__(self):
        self.buffer = ""

    def feed(self, text, final=False):
        self.buffer += text
        # Reject thought/protocol output before sending it to the speaker.
        if any(
            marker in self.buffer.lower()
            for marker in ("<think", "</think", "```", '{"text"')
        ):
            raise ValueError("Unexpected reasoning or protocol output")
        result = []
        while True:
            match = re.search(r'[.!?](?:["\u201d])?(?:\s+|$)', self.buffer)
            if not match or (match.end() == len(self.buffer) and not final):
                break
            phrase = self.buffer[: match.end()].strip()
            self.buffer = self.buffer[match.end() :]
            if phrase:
                result.append(phrase)
        if final and self.buffer.strip():
            result.append(self.buffer.strip())
            self.buffer = ""
        return result
