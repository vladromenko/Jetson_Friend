from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


class ObjectMemory:
    def __init__(self, path: Path):
        self.path = path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.items = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            self.items = {}

    @staticmethod
    def _key(label: str) -> str:
        return " ".join(label.lower().strip(" .?!,").split())

    def remember(self, label: str, location: str) -> None:
        key = self._key(label)
        location = location.strip()
        if not key or not location:
            return
        self.items.pop(key, None)
        self.items[key] = {"location": location, "seen_at": datetime.now().astimezone().isoformat(timespec="minutes")}
        self.items = dict(list(self.items.items())[-30:])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix="object_memory_", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.items, handle, ensure_ascii=False, indent=2)
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def recall(self, label: str) -> dict | None:
        item = self.items.get(self._key(label))
        return item if isinstance(item, dict) and item.get("location") else None
