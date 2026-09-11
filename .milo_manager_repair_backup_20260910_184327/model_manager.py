#!/usr/bin/env python3
import os
import signal
import subprocess
import time
from pathlib import Path

from model_registry import ModelRegistry

ROOT = Path(os.getenv("JETSON_FRIEND_ROOT", Path(__file__).resolve().parents[1])).resolve()
CONFIG = ROOT / "config.env"

class ModelManager:
    def __init__(self):
        self.registry = ModelRegistry()

    def _read_config(self):
        data = {}
        if CONFIG.exists():
            for raw in CONFIG.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                data[k] = v
        return data

    def _write_config(self, updates):
        lines = CONFIG.read_text(encoding="utf-8").splitlines() if CONFIG.exists() else []
        seen = set()
        out = []
        for raw in lines:
            stripped = raw.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k = stripped.split("=",1)[0]
                if k in updates:
                    if k not in seen:
                        out.append(f"{k}={updates[k]}")
                        seen.add(k)
                    continue
            out.append(raw)
        for k, v in updates.items():
            if k not in seen:
                out.append(f"{k}={v}")
        CONFIG.write_text("\n".join(out).rstrip()+"\n", encoding="utf-8")

    def select_text(self, model_id):
        info = self.registry.get(model_id)
        if info.get("type") != "text":
            raise ValueError("Not a text model")
        if not info["installed"]:
            raise FileNotFoundError("Model is not installed")
        path = info["paths"][0]
        self._write_config({
            "LLM_MODEL": path,
            "ACTIVE_TEXT_MODEL": model_id,
        })
        return info

    def select_vlm(self, model_id):
        info = self.registry.get(model_id)
        if info.get("type") != "vision":
            raise ValueError("Not a vision model")
        if not info["installed"]:
            raise FileNotFoundError("Model is not installed")
        paths = info["paths"]
        model_path = next((p for p in paths if "mmproj" not in p.lower()), None)
        mmproj = next((p for p in paths if "mmproj" in p.lower()), None)
        if not model_path or not mmproj:
            raise ValueError("VLM entry must contain model + mmproj")
        self._write_config({
            "VLM_MODEL": model_path,
            "VLM_MMPROJ": mmproj,
            "ACTIVE_VLM_MODEL": model_id,
            "VLM_ENABLE": "1",
        })
        return info

    def current(self):
        cfg = self._read_config()
        return {
            "text_model": cfg.get("ACTIVE_TEXT_MODEL", ""),
            "text_model_path": cfg.get("LLM_MODEL", ""),
            "vlm_model": cfg.get("ACTIVE_VLM_MODEL", ""),
            "vlm_model_path": cfg.get("VLM_MODEL", ""),
            "vlm_mmproj": cfg.get("VLM_MMPROJ", ""),
        }

    def milo_running(self):
        result = subprocess.run(["pgrep","-af","src/main.py"], text=True, capture_output=True)
        return result.returncode == 0 and bool(result.stdout.strip())

    def start_milo(self):
        if self.milo_running():
            return "already running"
        subprocess.Popen(
            ["bash", str(ROOT/"start.sh")],
            cwd=ROOT,
            stdout=open(ROOT/"milo_runtime.log","a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return "started"

    def stop_milo(self):
        subprocess.run(["pkill","-f","src/main.py"], check=False)
        time.sleep(1)
        return "stopped"

    def restart_milo(self):
        self.stop_milo()
        return self.start_milo()
