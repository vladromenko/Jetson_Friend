#!/usr/bin/env python3
from pathlib import Path
import datetime
import py_compile
import shutil
import subprocess
import sys

ROOT = Path.home() / "Jetson_Friend"
MANAGER = ROOT / "src" / "robotics" / "manager.py"
CONFIG = ROOT / "config.env"

def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

if not MANAGER.exists() or not CONFIG.exists():
    fail("Run this on the Jetson with ~/Jetson_Friend present.")

try:
    baseline = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "198afd0:src/robotics/manager.py"],
        text=True,
    )
except subprocess.CalledProcessError:
    fail("Cannot read manager.py from commit 198afd0.")

current = MANAGER.read_text()
if current != baseline:
    fail("src/robotics/manager.py is not the exact 198afd0 version. Nothing changed.")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup = ROOT / "data" / f"face_follow_patch_backup_{stamp}"
backup.mkdir(parents=True, exist_ok=False)
shutil.copy2(MANAGER, backup / "manager.py")
shutil.copy2(CONFIG, backup / "config.env")

def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        fail(f"{label}: expected exactly one match, found {count}. Backup is at {backup}")
    return text.replace(old, new, 1)

anchor = '        self.face_pose = self._pose("MILO_ARM_FACE_SEARCH_POSE", "90,115,115,110,135,120")\n'
insert = '''        # J1-only face following. J2..J6 stay at the operator-calibrated face pose.
        self.face_follow_deadband_x = float(os.getenv("MILO_FACE_FOLLOW_DEADBAND_X", "0.08"))
        self.face_follow_step = float(os.getenv("MILO_FACE_FOLLOW_STEP_DEG", "1"))
        self.face_follow_runtime_ms = int(os.getenv("MILO_FACE_FOLLOW_RUNTIME_MS", "320"))
        self.face_follow_settle = float(os.getenv("MILO_FACE_FOLLOW_SETTLE_SEC", "0.12"))
        self.face_follow_sign = int(os.getenv("MILO_FACE_FOLLOW_J1_SIGN", "1"))
        self.face_follow_reverse_margin = float(os.getenv("MILO_FACE_FOLLOW_REVERSE_MARGIN", "0.025"))
        if not 0.03 <= self.face_follow_deadband_x <= 0.20:
            raise ValueError("Face-follow deadband must be 0.03..0.20")
        if self.face_follow_step != 1:
            raise ValueError("Initial face-follow step is intentionally fixed at 1 degree")
        if not 320 <= self.face_follow_runtime_ms <= 1200:
            raise ValueError("Face-follow runtime must be 320..1200 ms")
        if not 0.10 <= self.face_follow_settle <= 1.0:
            raise ValueError("Face-follow settle must be 0.10..1.0 s")
        if self.face_follow_sign not in (-1, 1):
            raise ValueError("MILO_FACE_FOLLOW_J1_SIGN must be -1 or +1")
        if not 0.01 <= self.face_follow_reverse_margin <= 0.08:
            raise ValueError("Face-follow reverse margin must be 0.01..0.08")
        self.face_follow_sign_verified = False
        self.face_follow_previous_error = None
        self.face_follow_previous_delta = None

        self.face_pose = self._pose("MILO_ARM_FACE_SEARCH_POSE", "90,115,115,110,135,120")
'''
current = replace_once(current, anchor, insert, "follow settings")

old = '''                    self.pose_ready = False
                    self.face_search.reset()
                    print("[MILO ARM] SAFE: arm endpoint or fresh RGB-D lost; resume required", flush=True)
'''
new = '''                    self.pose_ready = False
                    self.face_search.reset()
                    self.face_follow_sign_verified = False
                    self.face_follow_previous_error = None
                    self.face_follow_previous_delta = None
                    print("[MILO ARM] SAFE: arm endpoint or fresh RGB-D lost; resume required", flush=True)
'''
current = replace_once(current, old, new, "SAFE reset")

old = '''                self.pose_ready = True
                self.face_search.reset()
                self.next_search_move = time.monotonic() + self.face_search.settle
                self.mode = "FACE_SEARCH"
'''
new = '''                self.pose_ready = True
                self.face_search.reset()
                self.face_follow_sign_verified = False
                self.face_follow_previous_error = None
                self.face_follow_previous_delta = None
                self.next_search_move = time.monotonic() + self.face_search.settle
                self.mode = "FACE_SEARCH"
'''
current = replace_once(current, old, new, "pose reset")

old = '''            previous = self.mode
            self.mode = self.face_search.observe(faces, stamp, time.monotonic())
            if self.mode != previous:
                print(f"[MILO ARM] {self.mode}: " +
                      ("face confirmed; no further search commands" if self.mode == "FACE_FOUND"
                       else "face lost after timeout; stale lock cleared"), flush=True)
            if self.mode == "FACE_FOUND" or self.face_search.hits:
                # Freeze even during acquisition. A pending short move may finish;
                # there is no measured-position hold/cancel primitive in this protocol.
                return
            if now < self.next_search_move:
                return
            target = self.face_search.next_target(self.state[1])
            msg = ArmJoint()
            msg.id, msg.joint, msg.time = 1, int(round(target)), self.face_search.runtime_ms
            if self.stop_request.is_set():
                return
            self.node.joint_pub.publish(msg)
            self.state[1] = msg.joint  # Commanded, NOT measured.
            self.next_search_move = time.monotonic() + msg.time / 1000 + self.face_search.settle
            print(f"[MILO ARM] FACE_SEARCH J1 command={msg.joint} runtime_ms={msg.time}", flush=True)
'''
new = '''            previous = self.mode
            observed_at = time.monotonic()
            self.mode = self.face_search.observe(faces, stamp, observed_at)
            if self.mode != previous:
                print(f"[MILO ARM] {self.mode}: " +
                      ("face confirmed; J1 follow enabled" if self.mode == "FACE_FOUND"
                       else "face lost after timeout; bounded reacquisition enabled"), flush=True)

            if self.mode == "FACE_FOUND":
                # FACE_FOUND can persist briefly after a missed detector frame.
                # Never move from a stale box: follow only a face matched THIS frame.
                if self.face_search.last_seen != observed_at or self.face_search.box is None:
                    return
                if now < self.next_search_move:
                    return

                x1, _, x2, _ = self.face_search.box
                cx = (x1 + x2) / 2.0
                error = cx - 0.5

                # Learn image-direction sign from the first 1-degree correction.
                # If it clearly makes same-side error worse, reverse once.
                if (not self.face_follow_sign_verified and
                        self.face_follow_previous_error is not None and
                        self.face_follow_previous_delta is not None):
                    old_error = self.face_follow_previous_error
                    same_side = error * old_error > 0
                    if same_side and abs(error) > abs(old_error) + self.face_follow_reverse_margin:
                        self.face_follow_sign *= -1
                        self.face_follow_sign_verified = True
                        print(f"[MILO ARM] FACE_FOLLOW learned opposite J1 sign: {self.face_follow_sign:+d}", flush=True)
                    elif (not same_side or
                          abs(error) < abs(old_error) - self.face_follow_reverse_margin):
                        self.face_follow_sign_verified = True
                        print(f"[MILO ARM] FACE_FOLLOW J1 sign verified: {self.face_follow_sign:+d}", flush=True)
                    self.face_follow_previous_error = None
                    self.face_follow_previous_delta = None

                if abs(error) <= self.face_follow_deadband_x:
                    return

                direction = 1 if error > 0 else -1
                delta = self.face_follow_sign * direction * self.face_follow_step
                target = max(self.face_search.minimum,
                             min(self.face_search.maximum, self.state[1] + delta))
                if target == self.state[1]:
                    return

                msg = ArmJoint()
                msg.id = 1
                msg.joint = int(round(target))
                msg.time = self.face_follow_runtime_ms
                if self.stop_request.is_set():
                    return
                self.node.joint_pub.publish(msg)
                self.face_follow_previous_error = error
                self.face_follow_previous_delta = delta
                self.state[1] = msg.joint  # Commanded, NOT measured.
                self.next_search_move = (
                    time.monotonic() + msg.time / 1000 + self.face_follow_settle
                )
                print(
                    f"[MILO ARM] FACE_FOLLOW cx={cx:.3f} err={error:+.3f} "
                    f"J1={msg.joint} sign={self.face_follow_sign:+d}",
                    flush=True,
                )
                return

            if self.face_search.hits:
                # During acquisition, freeze search until face confirmation.
                return

            self.face_follow_previous_error = None
            self.face_follow_previous_delta = None
            if now < self.next_search_move:
                return
            target = self.face_search.next_target(self.state[1])
            msg = ArmJoint()
            msg.id, msg.joint, msg.time = 1, int(round(target)), self.face_search.runtime_ms
            if self.stop_request.is_set():
                return
            self.node.joint_pub.publish(msg)
            self.state[1] = msg.joint  # Commanded, NOT measured.
            self.next_search_move = time.monotonic() + msg.time / 1000 + self.face_search.settle
            print(f"[MILO ARM] FACE_SEARCH J1 command={msg.joint} runtime_ms={msg.time}", flush=True)
'''
current = replace_once(current, old, new, "FACE_FOUND follow block")

old = '''            self.face_follow = True
            self.face_search.reset()
            self.mode = "FACE_SEARCH" if self.pose_ready else "FACE_SEARCH_POSE"
            return RobotResult(True, "Face search enabled. I will stop when I find a face.", "neutral")
'''
new = '''            self.face_follow = True
            self.face_search.reset()
            self.face_follow_sign_verified = False
            self.face_follow_previous_error = None
            self.face_follow_previous_delta = None
            self.mode = "FACE_SEARCH" if self.pose_ready else "FACE_SEARCH_POSE"
            return RobotResult(True, "Face tracking enabled.", "neutral")
'''
current = replace_once(current, old, new, "tracking resume")

MANAGER.write_text(current)

settings = {
    "MILO_FACE_FOLLOW_DEADBAND_X": "0.08",
    "MILO_FACE_FOLLOW_STEP_DEG": "1",
    "MILO_FACE_FOLLOW_RUNTIME_MS": "320",
    "MILO_FACE_FOLLOW_SETTLE_SEC": "0.12",
    "MILO_FACE_FOLLOW_J1_SIGN": "1",
    "MILO_FACE_FOLLOW_REVERSE_MARGIN": "0.025",
}
lines = CONFIG.read_text().splitlines()
for key, value in settings.items():
    for i, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
CONFIG.write_text("\n".join(lines) + "\n")

try:
    py_compile.compile(str(MANAGER), doraise=True)
except Exception as exc:
    shutil.copy2(backup / "manager.py", MANAGER)
    shutil.copy2(backup / "config.env", CONFIG)
    fail(f"Syntax validation failed and files were restored: {exc}")

print("PATCH OK")
print(f"Backup: {backup}")
for line in CONFIG.read_text().splitlines():
    if line.startswith("MILO_ARM_FACE_SEARCH_POSE="):
        print(line)
print("Face follow: J1 only, 1-degree steps, stale detections never command motion.")
print("J2/J3/J4/J5/J6 remain fixed at the saved face pose.")
