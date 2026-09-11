#!/usr/bin/env python3
import json
import os
from pathlib import Path

ROOT = Path(os.getenv("JETSON_FRIEND_ROOT", Path(__file__).resolve().parents[1])).resolve()
CATALOG = ROOT / "models" / "catalog.json"

class ModelRegistry:
    def __init__(self):
        self.root = ROOT
        self.catalog_path = CATALOG

    def load(self):
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return data.get("models", {})

    def installed(self, info):
        files = info.get("files", [])
        return bool(files) and all((self.root/"models"/f["target"]).is_file() for f in files)

    def list(self):
        models = self.load()
        result = []
        for model_id, info in models.items():
            item = dict(info)
            item["id"] = model_id
            item["installed"] = self.installed(info)
            item["paths"] = [str(self.root/"models"/f["target"]) for f in info.get("files", [])]
            result.append(item)
        return result

    def get(self, model_id):
        models = self.load()
        if model_id not in models:
            raise KeyError(model_id)
        info = dict(models[model_id])
        info["id"] = model_id
        info["installed"] = self.installed(info)
        info["paths"] = [str(self.root/"models"/f["target"]) for f in info.get("files", [])]
        return info
