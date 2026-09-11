"""Local timings only: never log prompts, names, embeddings or audio."""

import json
import os
from pathlib import Path
import threading
import time
import uuid

_LOCK = threading.Lock()


class TurnTiming:
    def __init__(self, start=None, source="keyboard"):
        self.start = time.monotonic() if start is None else start
        self.data = {"turn_id": str(uuid.uuid4()), "source": source}

    def mark(self, stage):
        self.data[stage + "_ms"] = round((time.monotonic() - self.start) * 1000, 2)

    def finish(self):
        self.mark("end_to_end")
        destination = os.getenv("MILO_METRICS_PATH", "")
        if destination:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK, path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(self.data, separators=(",", ":")) + "\n")
        return dict(self.data)
