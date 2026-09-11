#!/usr/bin/env python3
"""Inspect live perception and audio I/O without persisting frames or audio."""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from identity import IdentityManager
from memory import Memory
from speech import Speech
from vision import Vision


class Face:
    def set_gaze(self, *args):
        pass


def main():
    vision = Vision()
    worker = threading.Thread(target=vision.run, args=(Face(),), daemon=True)
    worker.start()
    result = {}
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if vision.get_scene().get("updated"):
                break
            time.sleep(0.2)
        time.sleep(5)
        scene = vision.get_scene()
        result["vision"] = {
            k: scene.get(k) for k in ("vision_backend", "faces", "metrics")
        }
        with tempfile.TemporaryDirectory() as tmp:
            previous_root = os.environ.get("JETSON_FRIEND_ROOT")
            os.environ["JETSON_FRIEND_ROOT"] = tmp
            try:
                identity = IdentityManager(
                    Memory(),
                    model_path=str(ROOT / "models/vision/face_recognition_sface.onnx"),
                )
                samples = []
                for _ in range(5):
                    crop, meta = vision.get_face_crop(
                        min_quality=identity.min_face_quality
                    )
                    if crop is not None:
                        start = time.monotonic()
                        embedding = identity.extract_embedding(
                            crop, meta.get("local_landmarks")
                        )
                        samples.append(
                            {
                                "ms": (time.monotonic() - start) * 1000,
                                "dimensions": len(embedding)
                                if embedding is not None
                                else None,
                                "quality": meta["quality"],
                            }
                        )
                    time.sleep(0.4)
                result["face_embeddings"] = samples
            finally:
                if previous_root:
                    os.environ["JETSON_FRIEND_ROOT"] = previous_root
                else:
                    os.environ.pop("JETSON_FRIEND_ROOT", None)
        speech = Speech()
        device = speech._find_input_device()
        result["input_device"] = device
        if device is not None:
            audio = sd.rec(
                16000, samplerate=16000, channels=1, device=device, dtype="float32"
            )
            sd.wait()
            result["microphone_rms"] = float(np.sqrt(np.mean(audio * audio)))
        result["output_device"] = speech._find_output_device()
        Path("artifacts/perception.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        vision.stop()
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
