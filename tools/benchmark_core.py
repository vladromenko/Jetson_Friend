#!/usr/bin/env python3
"""Controlled local prompts and synthetic audio. No user recordings or memory DB."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ai import HughAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    result = {
        "label": str(args.output),
        "context": os.getenv("LLAMA_CTX"),
        "gpu_layers": os.getenv("LLAMA_GPU_LAYERS"),
        "runs": [],
    }
    log = open(args.output + ".tegrastats", "w")
    stats = subprocess.Popen(
        ["tegrastats", "--interval", "1000"], stdout=log, stderr=subprocess.STDOUT
    )
    ai = None
    try:
        started = time.monotonic()
        ai = HughAI()
        result["load_ms"] = (time.monotonic() - started) * 1000
        for prompt in [
            "Hi Milo.",
            "I study robotics. Suggest one small project.",
            "What do I prefer to drink?",
        ]:
            started = time.monotonic()
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are MILO. Reply in one short spoken sentence. Relevant memory: the user prefers unsweetened coffee.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 96,
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
            }
            request = urllib.request.Request(
                ai.server_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            row = {
                "prompt": prompt,
                "content": "",
                "reasoning": "",
                "first_token_ms": None,
            }
            with urllib.request.urlopen(request, timeout=120) as response:
                for line in response:
                    if (
                        not line.startswith(b"data: ")
                        or line.strip() == b"data: [DONE]"
                    ):
                        continue
                    event = json.loads(line[6:])
                    if event.get("usage"):
                        row["usage"] = event["usage"]
                    if event.get("timings"):
                        row["timings"] = event["timings"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or ""
                        if (content or reasoning) and row["first_token_ms"] is None:
                            row["first_token_ms"] = (time.monotonic() - started) * 1000
                        row["content"] += content
                        row["reasoning"] += reasoning
            row["total_ms"] = (time.monotonic() - started) * 1000
            result["runs"].append(row)
        if not args.stream:
            started = time.monotonic()
            reply = ai.ask("Hello Milo. How are you?")
            result["app_json_ms"] = (time.monotonic() - started) * 1000
            result["app_json_reply"] = reply
        with tempfile.TemporaryDirectory() as tmp:
            wav = str(Path(tmp) / "synthetic.wav")
            started = time.monotonic()
            subprocess.run(
                [
                    os.environ["PIPER_BIN"],
                    "--model",
                    os.environ["PIPER_VOICE"],
                    "--output_file",
                    wav,
                ],
                input="Hello Milo. Please remember that my robotics demonstration is on Friday.",
                text=True,
                check=True,
                capture_output=True,
            )
            result["tts_ms"] = (time.monotonic() - started) * 1000
            result["whisper"] = []
            for model in ("ggml-base.en.bin", "ggml-small.en.bin"):
                path = ROOT / "models" / "whisper" / model
                if not path.exists():
                    continue
                started = time.monotonic()
                process = subprocess.run(
                    [
                        os.environ["WHISPER_BIN"],
                        "-m",
                        str(path),
                        "-f",
                        wav,
                        "-l",
                        "en",
                        "-nt",
                        "-np",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                result["whisper"].append(
                    {
                        "model": model,
                        "ms": (time.monotonic() - started) * 1000,
                        "code": process.returncode,
                        "text": process.stdout.strip(),
                        "backend": process.stderr[-2000:],
                    }
                )
    finally:
        if ai:
            ai.close()
        stats.terminate()
        stats.wait(timeout=5)
        log.close()
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
