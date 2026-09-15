#!/usr/bin/env python3
from pathlib import Path
import datetime
import shutil
import subprocess
import sys

ROOT = Path.home() / "Jetson_Friend"
CONFIG = ROOT / "config.env"
ONNX = ROOT / "models" / "vision" / "face_detection_yunet.onnx"

def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

if not CONFIG.exists():
    fail(f"Missing {CONFIG}")
if not ONNX.exists():
    fail(f"Missing YuNet ONNX model: {ONNX}")

# Validate that the same OpenCV/Python environment used by Milo can load the ONNX model
py = ROOT / ".venv" / "bin" / "python"
if not py.exists():
    py = Path(sys.executable)

probe = (
    "import cv2,sys;"
    f"p={str(ONNX)!r};"
    "d=cv2.FaceDetectorYN.create(p,'',(320,320),0.55,0.3,5000);"
    "print('OpenCV',cv2.__version__,'YuNet ONNX OK',p)"
)
r = subprocess.run([str(py), "-c", probe], text=True, capture_output=True)
if r.returncode != 0:
    print(r.stdout, end="")
    print(r.stderr, end="", file=sys.stderr)
    fail("YuNet ONNX could not be loaded; config was NOT changed.")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup = ROOT / "data" / f"config_before_arm_yunet_{stamp}.env"
backup.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(CONFIG, backup)

lines = CONFIG.read_text().splitlines()
updates = {
    "MILO_FACE_MODEL": str(ONNX),
    "MILO_FACE_YUNET_SCORE": "0.55",
    "MILO_FACE_IMAGE_ROTATION_DEG": "0",
    "MILO_ARM_FACE_SEARCH_POSE": "67,91,95,0,91,118",
}
for key, value in updates.items():
    for i, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")

CONFIG.write_text("\n".join(lines) + "\n")

print("ARM YUNET FIX OK")
print(r.stdout.strip())
print(f"Backup: {backup}")
print(f"MILO_FACE_MODEL={ONNX}")
print("The arm-camera manager will now use ONNX YuNet instead of the TensorRT .engine file.")
