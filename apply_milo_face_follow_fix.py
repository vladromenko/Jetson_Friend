#!/usr/bin/env python3
from pathlib import Path
import datetime
import shutil
import sys

ROOT = Path.home() / "Jetson_Friend"
MANAGER = ROOT / "src" / "robotics" / "manager.py"
CONFIG = ROOT / "config.env"

def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

if not MANAGER.exists() or not CONFIG.exists():
    fail("Expected ~/Jetson_Friend/src/robotics/manager.py and config.env")

text = MANAGER.read_text()

required = [
    '[MILO ARM] FACE_FOLLOW cx=',
    'self.face_follow_sign_verified',
    'self.face_follow_deadband_x',
]
missing = [x for x in required if x not in text]
if missing:
    fail("Expected previous face-follow patch is not present; nothing changed.")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup = ROOT / "data" / f"face_follow_fix_{stamp}"
backup.mkdir(parents=True, exist_ok=False)
shutil.copy2(MANAGER, backup / "manager.py")
shutil.copy2(CONFIG, backup / "config.env")

old = '        self.face_follow_deadband_x = float(os.getenv("MILO_FACE_FOLLOW_DEADBAND_X", "0.08"))\n'
new = (
    '        self.face_follow_deadband_x = float(os.getenv("MILO_FACE_FOLLOW_DEADBAND_X", "0.06"))\n'
    '        self.face_follow_target_x = float(os.getenv("MILO_FACE_FOLLOW_TARGET_X", "0.36"))\n'
)
if old in text:
    text = text.replace(old, new, 1)
elif 'self.face_follow_target_x' not in text:
    fail("Could not locate face-follow deadband setting; nothing changed.")

old = '''        if not 0.03 <= self.face_follow_deadband_x <= 0.20:
            raise ValueError("Face-follow deadband must be 0.03..0.20")
'''
new = '''        if not 0.03 <= self.face_follow_deadband_x <= 0.20:
            raise ValueError("Face-follow deadband must be 0.03..0.20")
        if not 0.20 <= self.face_follow_target_x <= 0.80:
            raise ValueError("MILO_FACE_FOLLOW_TARGET_X must be 0.20..0.80")
'''
if old in text:
    text = text.replace(old, new, 1)
elif 'MILO_FACE_FOLLOW_TARGET_X must be' not in text:
    fail("Could not locate face-follow validation; nothing changed.")

start_marker = '''                x1, _, x2, _ = self.face_search.box
                cx = (x1 + x2) / 2.0
                error = cx - 0.5
'''
start = text.find(start_marker)
if start < 0:
    if 'error = cx - self.face_follow_target_x' not in text:
        fail("Could not locate old FACE_FOLLOW control block; nothing changed.")
else:
    end_marker = '''                if abs(error) <= self.face_follow_deadband_x:
                    return
'''
    end = text.find(end_marker, start)
    if end < 0:
        fail("Could not locate FACE_FOLLOW deadband block; nothing changed.")
    end += len(end_marker)

    replacement = '''                x1, _, x2, _ = self.face_search.box
                cx = (x1 + x2) / 2.0

                # Calibrated from the physically verified DaBai frame.
                # The desired face position is x ~= 0.36, not the geometric image center.
                error = cx - self.face_follow_target_x

                # Fixed from observed real motion:
                # decreasing J1 moved the face right in image;
                # increasing J1 moved the face left.
                # So sign=+1 is correct. Never auto-flip from one noisy detection.
                self.face_follow_sign = 1
                self.face_follow_sign_verified = True
                self.face_follow_previous_error = None
                self.face_follow_previous_delta = None

                if abs(error) <= self.face_follow_deadband_x:
                    if self.mode != "FACE_HOLD":
                        print(
                            f"[MILO ARM] FACE_HOLD cx={cx:.3f} target={self.face_follow_target_x:.3f} "
                            f"J1={int(round(self.state[1]))}",
                            flush=True,
                        )
                    self.mode = "FACE_HOLD"
                    return

                self.mode = "FACE_FOUND"
'''
    text = text[:start] + replacement + text[end:]

old = '                    self.mode in {"FACE_SEARCH_POSE", "FACE_SEARCH", "FACE_FOUND"}):\n'
new = '                    self.mode in {"FACE_SEARCH_POSE", "FACE_SEARCH", "FACE_FOUND", "FACE_HOLD"}):\n'
if old in text:
    text = text.replace(old, new, 1)
elif '"FACE_HOLD"' not in text:
    fail("Could not extend face-loop modes; nothing changed.")

MANAGER.write_text(text)

settings = {
    "MILO_FACE_FOLLOW_TARGET_X": "0.36",
    "MILO_FACE_FOLLOW_DEADBAND_X": "0.06",
    "MILO_FACE_FOLLOW_J1_SIGN": "1",
    "MILO_FACE_SEARCH_STEP_DEG": "1",
    "MILO_FACE_SEARCH_RUNTIME_MS": "420",
    "MILO_FACE_SEARCH_SETTLE_SEC": "0.18",
    "MILO_FACE_SEARCH_CONFIRMATIONS": "3",
    "MILO_FACE_SEARCH_LOSS_TIMEOUT_SEC": "2.5",
}
lines = CONFIG.read_text().splitlines()
for key, value in settings.items():
    for i, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")

for expected in (
    "MILO_ARM_FACE_SEARCH_POSE=67,91,95,0,91,118",
    "MILO_FACE_IMAGE_ROTATION_DEG=0",
):
    key = expected.split("=", 1)[0]
    found = next((x for x in lines if x.startswith(key + "=")), None)
    if found != expected:
        shutil.copy2(backup / "manager.py", MANAGER)
        shutil.copy2(backup / "config.env", CONFIG)
        fail(f"Safety check failed: expected {expected}, found {found}; restored backup")

CONFIG.write_text("\n".join(lines) + "\n")

try:
    py_compile.compile(str(MANAGER), doraise=True)
except Exception as exc:
    shutil.copy2(backup / "manager.py", MANAGER)
    shutil.copy2(backup / "config.env", CONFIG)
    fail(f"Syntax check failed; restored backup: {exc}")

print("FACE FOLLOW FIX OK")
print(f"Backup: {backup}")
print("target_x=0.36, deadband=0.06")
print("J1 direction fixed to +1 from real observed motion; auto sign-flip removed")
print("search step=1 degree, runtime=420 ms, settle=180 ms")
print("verified pose preserved: 67,91,95,0,91,118")
