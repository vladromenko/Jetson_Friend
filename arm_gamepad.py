#!/usr/bin/env python3
"""Standalone guarded gamepad control with RGB-D demonstration recording."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

from milo.config import Config
from milo.ros_runtime import RosRuntime
from milo.safety import ArmSafetyGate, SafetyResult


ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "data" / "candy_demos"
JOINTS = (1, 2, 3, 4, 5)
FEEDBACK_JOINTS = (1, 2, 3, 4, 5, 6)
LIMITS = {1: (0, 180), 2: (90, 115), 3: (0, 180),
          4: (0, 180), 5: (0, 270)}
MANUAL_RUNTIME_MS = 180
MANUAL_MAX_STEP_DEG = 5
MANUAL_MAX_SPEED_DEG_S = 45
TELEOP_AXIS_DEADBAND = 0.16
TELEOP_AXIS_EXPO = 1.35
TELEOP_AXIS_ALPHA = 0.42
TELEOP_ACCEL_DEG_S2 = 70.0
TELEOP_BRAKE_DEG_S2 = 130.0
TELEOP_MAX_COMMAND_STEP_DEG = 4
FIELDS = ["step", "t", "state_source", *(f"j{i}" for i in FEEDBACK_JOINTS),
          *(f"a{i}" for i in JOINTS), "rgb", "depth"]


def direction(value: float, deadband: float = 0.30) -> int:
    return 0 if abs(value) < deadband else (1 if value > 0 else -1)


def scaled_axis(value: float, deadband: float = TELEOP_AXIS_DEADBAND,
                expo: float = TELEOP_AXIS_EXPO) -> float:
    """Map a raw joystick axis to a smooth velocity request in -1..1."""
    if not math.isfinite(value) or abs(value) <= deadband:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    scaled = (abs(value) - deadband) / (1.0 - deadband)
    return sign * min(1.0, scaled) ** expo


def requested_velocities(pad) -> dict[int, float]:
    axis = lambda i: pad.get_axis(i) if i < pad.get_numaxes() else 0.0
    button = lambda i: i < pad.get_numbuttons() and bool(pad.get_button(i))

    moves = {
        1: scaled_axis(-axis(0)),
        2: scaled_axis(-axis(1)),
        3: scaled_axis(-axis(3)),
        4: scaled_axis(axis(2)),
        5: float(button(7)) - float(button(6)),
    }
    return {joint: value for joint, value in moves.items() if abs(value) > 1e-6}


def requested_moves(pad) -> dict[int, int]:
    return {joint: direction(value, 0.0)
            for joint, value in requested_velocities(pad).items()}


class SpeedControl:
    def __init__(self, base_speed: int = 20, minimum: int = 5, maximum: int = 35):
        self.speed = base_speed
        self.minimum = minimum
        self.maximum = maximum

    def adjust(self, amount: int) -> None:
        old = self.speed
        self.speed = max(self.minimum, min(self.maximum, self.speed + amount))
        if self.speed != old:
            print(
                f"[SPEED] {self.speed} deg/s "
                f"(step={self.command_step()} deg, runtime={self.command_runtime_ms()} ms)",
                flush=True,
            )

    def command_runtime_ms(self) -> int:
        return MANUAL_RUNTIME_MS

    def command_step(self) -> int:
        step = round(self.speed * self.command_runtime_ms() / 1000.0)
        return max(1, min(TELEOP_MAX_COMMAND_STEP_DEG, int(step)))


class SmoothTeleop:
    """Velocity-style teleop built on top of safe absolute servo commands."""

    def __init__(self, speed: SpeedControl, now: float | None = None):
        self.speed = speed
        self.filtered = {joint: 0.0 for joint in JOINTS}
        self.velocity = {joint: 0.0 for joint in JOINTS}
        self.residual = {joint: 0.0 for joint in JOINTS}
        self.last = time.monotonic() if now is None else now

    def reset(self, now: float | None = None) -> None:
        self.filtered = {joint: 0.0 for joint in JOINTS}
        self.velocity = {joint: 0.0 for joint in JOINTS}
        self.residual = {joint: 0.0 for joint in JOINTS}
        self.last = time.monotonic() if now is None else now

    def clear_joint(self, joint: int) -> None:
        if joint in self.residual:
            self.residual[joint] = 0.0

    def command_deltas(self, requested: dict[int, float],
                       now: float | None = None) -> dict[int, int]:
        now = time.monotonic() if now is None else now
        dt = max(0.02, min(0.35, now - self.last))
        self.last = now
        deltas: dict[int, int] = {}
        for joint in JOINTS:
            raw = max(-1.0, min(1.0, requested.get(joint, 0.0)))
            self.filtered[joint] += (raw - self.filtered[joint]) * TELEOP_AXIS_ALPHA
            target_velocity = self.filtered[joint] * self.speed.speed
            accel = TELEOP_BRAKE_DEG_S2 if abs(target_velocity) < abs(self.velocity[joint]) else TELEOP_ACCEL_DEG_S2
            dv = target_velocity - self.velocity[joint]
            limit = accel * dt
            if abs(dv) > limit:
                dv = math.copysign(limit, dv)
            self.velocity[joint] += dv
            if abs(raw) < 1e-6 and abs(self.filtered[joint]) < 0.04 and abs(self.velocity[joint]) < 0.5:
                self.filtered[joint] = 0.0
                self.velocity[joint] = 0.0
                self.residual[joint] = 0.0
                continue

            delta = self.velocity[joint] * dt + self.residual[joint]
            step = math.trunc(delta)
            self.residual[joint] = delta - step
            if abs(step) > TELEOP_MAX_COMMAND_STEP_DEG:
                step = int(math.copysign(TELEOP_MAX_COMMAND_STEP_DEG, step))
                self.residual[joint] = 0.0
            if step:
                deltas[joint] = int(step)
        return deltas


def apply_speed_event(event, speed: SpeedControl) -> bool:
    import pygame
    if event.type == pygame.JOYHATMOTION:
        y = int(event.value[1]) if len(event.value) > 1 else 0
        if y > 0:
            speed.adjust(5)
            return True
        if y < 0:
            speed.adjust(-5)
            return True
    if event.type == pygame.JOYBUTTONDOWN:
        if event.button == 12:
            speed.adjust(5)
            return True
        if event.button == 13:
            speed.adjust(-5)
            return True
    if event.type == pygame.KEYDOWN:
        plus_keys = {pygame.K_UP, pygame.K_EQUALS, getattr(pygame, "K_PLUS", pygame.K_EQUALS)}
        minus_keys = {pygame.K_DOWN, pygame.K_MINUS, getattr(pygame, "K_UNDERSCORE", pygame.K_MINUS)}
        if event.key in plus_keys:
            speed.adjust(5)
            return True
        if event.key in minus_keys:
            speed.adjust(-5)
            return True
    return False


class ManualSafetyGate(ArmSafetyGate):
    def joint_ready(self, joint: int, now: float | None = None) -> tuple[bool, str]:
        now = time.monotonic() if now is None else now
        if joint not in JOINTS:
            return False, f"J{joint} is not allowed"
        if joint not in self.anchor:
            return False, self.reasons.get(joint, "waiting for stable feedback")
        if now - self.measured_stamp.get(joint, 0.0) > self.cfg.feedback_max_age:
            return False, f"J{joint} feedback stale"
        reason = self.reasons.get(joint, "")
        return (not reason), reason

    def validate(self, joint: int, target: float, runtime_ms: int,
                 now: float | None = None) -> SafetyResult:
        if joint == 6:
            return SafetyResult(False, "J6 is locked")
        return super().validate(joint, target, runtime_ms, now)


class ManualRuntime(RosRuntime):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.safety = ManualSafetyGate(cfg)
        self.latest_feedback: dict[int, int] = {}

    def command_manual(self, joint: int, target: int, runtime_ms: int | None = None) -> tuple[bool, str]:
        if joint == 6:
            return False, "J6 is locked"
        if not self.node or self.node.pub.get_subscription_count() < 1:
            return False, "STM32 command subscriber unavailable"
        if self.arm_busy():
            return False, "previous motion is still pending"
        runtime_ms = self.cfg.runtime_ms if runtime_ms is None else runtime_ms
        result = self.safety.validate(joint, target, runtime_ms)
        if not result.ok:
            return False, result.reason
        from arm_msgs.msg import ArmJoint

        msg = ArmJoint()
        msg.id = joint
        msg.joint = int(result.target)
        msg.time = runtime_ms
        before = self.safety.measured.get(joint)
        self.node.pub.publish(msg)
        self.safety.note_command(joint, int(result.target))
        now = time.monotonic()
        self._pending_motion[joint] = {
            "before": before,
            "target": int(result.target),
            "min_complete": now + runtime_ms / 1000.0,
            "deadline": now + max(1.5, runtime_ms / 1000.0 + 0.8),
            "movement_logged": False,
        }
        print(f"[ARM] J{joint}: {before} -> {result.target} ({runtime_ms} ms)", flush=True)
        return True, ""


class Episode:
    def __init__(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = DEMO_DIR / f"episode_{stamp}_{time.time_ns() % 1_000_000_000:09d}"
        (self.path / "rgb").mkdir(parents=True)
        (self.path / "depth").mkdir()
        self.file = (self.path / "trajectory.csv").open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(FIELDS)
        self.count = 0
        self.start = time.monotonic()

    def add(self, rgb, depth, state: dict[int, int], action: dict[int, int]) -> bool:
        import cv2

        basename = f"{self.count:06d}"
        rgb_name, depth_name = basename + ".jpg", basename + ".png"
        if not cv2.imwrite(str(self.path / "rgb" / rgb_name), rgb,
                           [cv2.IMWRITE_JPEG_QUALITY, 90]):
            return False
        if not cv2.imwrite(str(self.path / "depth" / depth_name), depth):
            return False
        self.writer.writerow([self.count, round(time.monotonic() - self.start, 3),
                              "measured", *(state.get(j, "") for j in FEEDBACK_JOINTS),
                              *(action.get(j, "") for j in JOINTS), rgb_name, depth_name])
        self.file.flush()
        self.count += 1
        return True

    def finish(self, success: bool) -> Path:
        self.file.close()
        meta = {"task": "manual candy demonstration", "success": success,
                "frames": self.count, "controlled_joints": list(JOINTS),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return self.path


def main() -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("[BLOCKED] Controller not found on Jetson", flush=True)
        return 1
    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"[READY] {pad.get_name()}: {pad.get_numaxes()} axes, {pad.get_numbuttons()} buttons", flush=True)
    cfg = Config.load(ROOT)
    cfg = replace(cfg, allowed_joints=JOINTS, hard_limits=LIMITS,
                  local_spans={j: 270 for j in JOINTS},
                  runtime_ms=MANUAL_RUNTIME_MS,
                  max_step_deg=MANUAL_MAX_STEP_DEG,
                  max_speed_deg_s=MANUAL_MAX_SPEED_DEG_S)
    runtime = ManualRuntime(cfg)
    armed = False
    episode = None
    speed = SpeedControl()
    teleop = SmoothTeleop(speed)
    last_capture = 0.0
    turn = 0
    last_status = 0.0
    print("START arm | SELECT stop | A record | B save | X unsuccessful | Ctrl+C quit", flush=True)
    print("Smooth velocity teleop: LX J1 | LY J2 (90..115) | RY J3 | RX J4 | L1/R1 J5 | J6 LOCKED", flush=True)
    print("D-pad up/down changes maximum speed by +/-5 deg/s", flush=True)
    try:
        runtime.start()
        if not runtime.available:
            return 1
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if runtime.node.pub.get_subscription_count() and runtime.safety.ready_joints():
                break
            time.sleep(0.1)
        else:
            print("[BLOCKED] STM32 or stable servo feedback unavailable", flush=True)
            return 1
        while True:
            now = time.monotonic()
            for event in pygame.event.get():
                if apply_speed_event(event, speed):
                    continue
                if event.type != pygame.JOYBUTTONDOWN:
                    continue
                if event.button == 11:
                    armed = True
                    teleop.reset(now)
                    print("[ARM] enabled", flush=True)
                elif event.button == 10:
                    armed = False
                    teleop.reset(now)
                    print("[ARM] stopped", flush=True)
                elif event.button == 0 and episode is None:
                    if (runtime.latest_color is None or runtime.latest_depth is None
                            or not runtime.camera_fresh(now)
                            or now - runtime.depth_stamp > cfg.camera_max_age):
                        print("[REC] DaBai color and depth are needed", flush=True)
                    else:
                        episode = Episode()
                        print(f"[REC] {episode.path}", flush=True)
                elif event.button == 1 and episode is not None:
                    path = episode.finish(True)
                    print(f"[SAVED] {path} ({episode.count} frames)", flush=True)
                    episode = None
                elif event.button == 3 and episode is not None:
                    path = episode.finish(False)
                    print(f"[UNSUCCESSFUL] {path} kept for review", flush=True)
                    episode = None

            action: dict[int, int] = {}
            if armed and not runtime.arm_busy():
                moves = teleop.command_deltas(requested_velocities(pad), now)
                for offset in range(len(JOINTS)):
                    joint = JOINTS[(turn + offset) % len(JOINTS)]
                    delta = moves.get(joint, 0)
                    if not delta or not runtime.safety.joint_ready(joint)[0]:
                        continue
                    current = runtime.safety.trusted[joint]
                    lo, hi = LIMITS[joint]
                    target = max(lo, min(hi, current + delta))
                    duration = speed.command_runtime_ms()
                    if target == current:
                        teleop.clear_joint(joint)
                        continue
                    ok, reason = runtime.command_manual(joint, target, duration)
                    if ok:
                        action[joint] = target
                        turn = joint % len(JOINTS)
                    else:
                        print(f"[ARM] J{joint}: {reason}", flush=True)
                    break

            if episode is not None and now - last_capture >= 0.1:
                rgb, depth = runtime.latest_color, runtime.latest_depth
                if (rgb is not None and depth is not None and runtime.camera_fresh(now)
                        and now - runtime.depth_stamp <= cfg.camera_max_age):
                    state = {j: runtime.latest_feedback[j] for j in FEEDBACK_JOINTS
                             if j in runtime.latest_feedback and
                             now - runtime.feedback_stamp <= cfg.feedback_max_age}
                    if len(state) == len(FEEDBACK_JOINTS) and not episode.add(rgb.copy(), depth.copy(), state, action):
                        print("[REC] image write failed; episode stopped", flush=True)
                        episode.finish(False)
                        episode = None
                    last_capture = now
            if now - last_status > 5:
                print("[STATUS] " + ("ARMED" if armed else "DISARMED") + " | "
                      + runtime.safety.status_line(), flush=True)
                last_status = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        return 0
    finally:
        if episode is not None:
            path = episode.finish(False)
            print(f"[REC] unfinished episode kept: {path}", flush=True)
        runtime.close()
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
