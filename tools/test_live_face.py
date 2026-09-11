#!/usr/bin/env python3
"""Short live identity test. Temporary local profile; no saved frames or vectors."""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from identity import IdentityManager
from memory import Memory
from vision import Vision


class Face:
    def set_gaze(self, *_):
        pass


def main():
    vision = Vision()
    worker = threading.Thread(target=vision.run, args=(Face(),), daemon=True)
    worker.start()
    report = {
        "enrollment": [],
        "validation": [],
        "rejected_frames": 0,
        "multiple_faces": 0,
        "no_face": 0,
        "samples": 0,
    }
    previous_root = os.environ.get("JETSON_FRIEND_ROOT")
    try:
        with tempfile.TemporaryDirectory(prefix="milo-live-face-") as tmp:
            os.environ["JETSON_FRIEND_ROOT"] = tmp
            identity = IdentityManager(
                Memory(),
                model_path=str(ROOT / "models/vision/face_recognition_sface.onnx"),
            )
            pid = identity.memory.create_person(
                "Temporary live test", make_current=False
            )
            deadline = time.monotonic() + 45
            enrollment_end = time.monotonic() + 25
            last_capture = None
            enrollment_track = None
            next_status = 0
            while time.monotonic() < deadline:
                scene = vision.get_scene()
                if scene.get("faces", 0) > 1:
                    report["multiple_faces"] += 1
                elif scene.get("faces", 0) == 0:
                    report["no_face"] += 1
                crop, meta = vision.get_face_crop(min_quality=0)
                if (
                    crop is not None
                    and meta is not None
                    and meta["captured"] != last_capture
                ):
                    last_capture = meta["captured"]
                    if meta["quality"] < identity.min_face_quality:
                        report["rejected_frames"] += 1
                    else:
                        start = time.monotonic()
                        vector = identity.extract_embedding(
                            crop, meta.get("local_landmarks")
                        )
                        elapsed = (time.monotonic() - start) * 1000
                        if vector is not None:
                            if time.monotonic() < enrollment_end:
                                if enrollment_track is None:
                                    enrollment_track = meta["track_id"]
                                if (
                                    meta["track_id"] == enrollment_track
                                    and identity.embedding_count(pid) < 5
                                ):
                                    added = identity.add_embedding(
                                        pid,
                                        vector,
                                        quality=meta["quality"],
                                        source="temporary_live_test",
                                    )
                                    report["enrollment"].append(
                                        {
                                            "quality": meta["quality"],
                                            "embedding_ms": round(elapsed, 2),
                                            "dimensions": len(vector),
                                            "added": added,
                                        }
                                    )
                            else:
                                start = time.monotonic()
                                result = identity.recognize_embedding(vector)
                                report["validation"].append(
                                    {
                                        "quality": meta["quality"],
                                        "embedding_ms": round(elapsed, 2),
                                        "match_ms": round(
                                            (time.monotonic() - start) * 1000, 2
                                        ),
                                        "status": result.status,
                                        "similarity": round(result.similarity, 4),
                                        "same_profile": result.person_id == pid,
                                    }
                                )
                if time.monotonic() >= next_status:
                    report["samples"] = identity.embedding_count(pid)
                    print(
                        json.dumps(
                            {
                                "phase": "enroll"
                                if time.monotonic() < enrollment_end
                                else "validate",
                                "faces": scene.get("faces", 0),
                                "quality": meta.get("quality") if meta else None,
                                "samples": report["samples"],
                                "validation_frames": len(report["validation"]),
                            }
                        ),
                        flush=True,
                    )
                    next_status = time.monotonic() + 5
                time.sleep(0.2)
            report["samples"] = identity.embedding_count(pid)
            report["vision"] = vision.get_scene().get("metrics")
            identity.memory.forget_person(pid)
    finally:
        vision.stop()
        worker.join(timeout=5)
        if previous_root:
            os.environ["JETSON_FRIEND_ROOT"] = previous_root
        else:
            os.environ.pop("JETSON_FRIEND_ROOT", None)
        path = ROOT / "artifacts" / "live_face_test.json"
        path.write_text(json.dumps(report, indent=2))
        print("Live face test finished. Temporary profile removed.", flush=True)


if __name__ == "__main__":
    main()
