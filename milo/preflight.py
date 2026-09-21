from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .audio import VoiceIO
from .llm import LocalModel
from .vision import VisionEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--audio", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = Config.load(root)

    if args.audio:
        voice = VoiceIO(cfg)
        if voice.sd is None or voice.input_device is None:
            raise RuntimeError("no usable sounddevice microphone was selected")
        print(f"[PREFLIGHT] microphone: {voice._input_name()}", flush=True)
        print(f"[PREFLIGHT] speaker: {voice.output_device}", flush=True)

    if args.vision:
        # RC3 only checked whether the detector object could be constructed. RC4
        # also verifies that the same Python runtime used by MILO receives a real
        # DaBai frame through rclpy.
        from .ros_runtime import RosRuntime
        ros = RosRuntime(cfg)
        vision = VisionEngine(cfg)
        try:
            ros.start()
            if not ros.wait_for_color(8.0):
                raise RuntimeError(
                    "MILO Python runtime did not receive a DaBai color frame. "
                    "Shell-level topic presence is not sufficient."
                )
            frame = ros.latest_color
            obs = vision.detect_face(frame)
            print(f"[PREFLIGHT] real DaBai frame: shape={frame.shape}", flush=True)
            print(f"[PREFLIGHT] vision backend: {vision.backend}", flush=True)
            if obs.center is None:
                print("[PREFLIGHT] face: none in current frame (detector executed successfully)", flush=True)
            else:
                print(
                    f"[PREFLIGHT] face: x={obs.center[0]:.3f} y={obs.center[1]:.3f} score={obs.score:.3f}",
                    flush=True,
                )
        finally:
            vision.close()
            ros.close()

    if args.llm or args.vlm:
        model = LocalModel(cfg.llm_url, str(cfg.llm_model), 24, cfg.llm_timeout)
        if args.llm:
            answer = model.health_check()
            print(f"[PREFLIGHT] LLM answer: {answer}", flush=True)
        if args.vlm:
            import cv2
            import numpy as np
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise RuntimeError("could not build VLM health-check image")
            answer = model.describe("Confirm that image input was accepted. Reply briefly.", encoded.tobytes())
            if not answer:
                raise RuntimeError("VLM returned an empty health-check response")
            print(f"[PREFLIGHT] VLM answer: {answer}", flush=True)


if __name__ == "__main__":
    main()
