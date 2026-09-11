#!/usr/bin/env python3

import os
import subprocess
import time
from pathlib import Path

from model_registry import ModelRegistry


ROOT = Path(
    os.getenv(
        "JETSON_FRIEND_ROOT",
        Path(__file__).resolve().parents[1],
    )
).resolve()

CONFIG = ROOT / "config.env"


class PanelModelManager:
    def __init__(self):
        self.registry = ModelRegistry()

    def _read_config(self):
        data = {}

        if not CONFIG.exists():
            return data

        for raw in CONFIG.read_text(
            encoding="utf-8"
        ).splitlines():
            line = raw.strip()

            if (
                line
                and not line.startswith("#")
                and "=" in line
            ):
                key, value = line.split("=", 1)
                data[key] = value

        return data

    def _write_config(self, updates):
        if CONFIG.exists():
            lines = CONFIG.read_text(
                encoding="utf-8"
            ).splitlines()
        else:
            lines = []

        seen = set()
        output = []

        for raw in lines:
            stripped = raw.strip()
            key = None

            if (
                stripped
                and not stripped.startswith("#")
                and "=" in stripped
            ):
                key = stripped.split("=", 1)[0]

            if key in updates:
                if key not in seen:
                    output.append(
                        f"{key}={updates[key]}"
                    )
                    seen.add(key)
            else:
                output.append(raw)

        for key, value in updates.items():
            if key not in seen:
                output.append(
                    f"{key}={value}"
                )

        CONFIG.write_text(
            "\n".join(output).rstrip() + "\n",
            encoding="utf-8",
        )

    def current(self):
        config = self._read_config()

        return {
            "text_model": config.get(
                "ACTIVE_TEXT_MODEL",
                "",
            ),
            "text_model_path": config.get(
                "LLM_MODEL",
                "",
            ),
            "vlm_model": config.get(
                "ACTIVE_VLM_MODEL",
                "",
            ),
            "vlm_model_path": config.get(
                "VLM_MODEL",
                "",
            ),
            "vlm_mmproj": config.get(
                "VLM_MMPROJ",
                "",
            ),
        }

    def select_text(self, model_id):
        info = self.registry.get(model_id)

        if info.get("type") != "text":
            raise ValueError(
                "Not a text model"
            )

        if not info["installed"]:
            raise FileNotFoundError(
                "Model is not installed"
            )

        model_path = info["paths"][0]

        self._write_config(
            {
                "LLM_MODEL": model_path,
                "ACTIVE_TEXT_MODEL": model_id,
            }
        )

        return info

    def select_vlm(self, model_id):
        info = self.registry.get(model_id)

        if info.get("type") != "vision":
            raise ValueError(
                "Not a vision model"
            )

        if not info["installed"]:
            raise FileNotFoundError(
                "Model is not installed"
            )

        model_path = None
        mmproj = None

        for path in info["paths"]:
            if "mmproj" in path.lower():
                mmproj = path
            elif model_path is None:
                model_path = path

        if not model_path or not mmproj:
            raise ValueError(
                "VLM entry must contain model + mmproj"
            )

        self._write_config(
            {
                "VLM_MODEL": model_path,
                "VLM_MMPROJ": mmproj,
                "ACTIVE_VLM_MODEL": model_id,
                "VLM_ENABLE": "1",
            }
        )

        return info

    def milo_running(self):
        result = subprocess.run(
            [
                "pgrep",
                "-af",
                "src/main.py",
            ],
            text=True,
            capture_output=True,
        )

        return (
            result.returncode == 0
            and bool(result.stdout.strip())
        )

    def start_milo(self):
        if self.milo_running():
            return "already running"

        log_path = ROOT / "milo_runtime.log"

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as log_file:
            subprocess.Popen(
                [
                    "bash",
                    str(ROOT / "start.sh"),
                ],
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        return "started"

    def stop_milo(self):
        subprocess.run(
            [
                "pkill",
                "-f",
                "src/main.py",
            ],
            check=False,
        )

        return "stopped"

    def restart_milo(self):
        self.stop_milo()
        time.sleep(1)
        return self.start_milo()
