from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class SafetyResult:
    ok: bool
    reason: str = ""
    target: int | None = None


class ArmSafetyGate:
    """Per-joint safety gate.

    Bad or out-of-range feedback from one servo must never disable unrelated
    safe axes. Each joint becomes usable only after fresh, stable measurements.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.measured: dict[int, int] = {}
        self.measured_stamp: dict[int, float] = {}
        self.anchor: dict[int, int] = {}
        self.local_limits: dict[int, tuple[int, int]] = {}
        self.trusted: dict[int, int] = {}
        self.last_command_stamp: dict[int, float] = {}
        self.reasons: dict[int, str] = {j: "waiting for stable feedback" for j in cfg.allowed_joints}
        self.history = {
            j: deque(maxlen=max(1, cfg.feedback_stable_samples)) for j in cfg.allowed_joints
        }

    def _range_ok(self, joint: int, value: int) -> bool:
        lo, hi = self.cfg.hard_limits[joint]
        return lo <= value <= hi

    def _normalize_feedback(self, joint: int, value: int) -> tuple[int | None, str | None]:
        lo, hi = self.cfg.hard_limits[joint]
        soft = max(0, int(getattr(self.cfg, "feedback_soft_limit_deg", 0)))
        if lo <= value <= hi:
            return value, None
        if lo - soft <= value < lo:
            return lo, f"J{joint} feedback {value} treated as calibrated lower stop {lo}"
        if hi < value <= hi + soft:
            return hi, f"J{joint} feedback {value} treated as calibrated upper stop {hi}"
        return None, f"J{joint} feedback {value} outside command range {lo}..{hi}"

    def update_measured(self, values: dict[int, int], stamp: float | None = None) -> None:
        now = time.monotonic() if stamp is None else stamp
        for joint in self.cfg.allowed_joints:
            if joint not in values:
                self.reasons[joint] = f"J{joint} feedback missing"
                continue

            raw_value = int(values[joint])
            normalized, note = self._normalize_feedback(joint, raw_value)
            if normalized is None:
                self.reasons[joint] = note or f"J{joint} feedback invalid"
                self.history[joint].clear()
                continue
            value = normalized

            previous = self.measured.get(joint)
            if previous is not None and abs(value - previous) > self.cfg.feedback_max_jump_deg:
                self.reasons[joint] = (
                    f"J{joint} feedback jump {abs(value - previous):.1f} deg exceeds "
                    f"{self.cfg.feedback_max_jump_deg:.1f}"
                )
                self.history[joint].clear()
                continue

            self.measured[joint] = value
            self.measured_stamp[joint] = now
            self.history[joint].append(value)

            if joint not in self.anchor:
                samples = list(self.history[joint])
                if len(samples) >= self.history[joint].maxlen:
                    if max(samples) - min(samples) <= self.cfg.feedback_stability_deg:
                        center = int(round(statistics.median(samples)))
                        hard_lo, hard_hi = self.cfg.hard_limits[joint]
                        span = self.cfg.local_spans[joint]
                        self.anchor[joint] = center
                        self.local_limits[joint] = (
                            max(hard_lo, center - span),
                            min(hard_hi, center + span),
                        )
                        self.trusted[joint] = center
                        self.reasons[joint] = ""
                    else:
                        self.reasons[joint] = f"J{joint} feedback not stable yet"
            else:
                # Reconcile the trusted command state only when the servo has had
                # time to move or is already close to our last command.
                trusted = self.trusted[joint]
                last_cmd = self.last_command_stamp.get(joint, 0.0)
                if abs(value - trusted) <= 1 or now - last_cmd > 1.5:
                    self.trusted[joint] = value
                self.reasons[joint] = ""

    def joint_ready(self, joint: int, now: float | None = None) -> tuple[bool, str]:
        now = time.monotonic() if now is None else now
        if joint not in self.cfg.allowed_joints or joint in {2, 6}:
            return False, f"J{joint} is not allowed"
        if joint not in self.anchor:
            return False, self.reasons.get(joint, f"J{joint} not initialized")
        stamp = self.measured_stamp.get(joint, 0.0)
        if now - stamp > self.cfg.feedback_max_age:
            return False, f"J{joint} feedback stale"
        if self.reasons.get(joint):
            return False, self.reasons[joint]
        return True, ""

    def ready_joints(self, now: float | None = None) -> set[int]:
        return {j for j in self.cfg.allowed_joints if self.joint_ready(j, now)[0]}

    def trusted_positions(self) -> dict[int, int]:
        return dict(self.trusted)

    def feedback_valid(self, now: float | None = None) -> tuple[bool, str]:
        """Compatibility helper: at least one tracking axis is usable."""
        ready = self.ready_joints(now)
        if ready:
            return True, ""
        reasons = [self.reasons.get(j, f"J{j} unavailable") for j in self.cfg.allowed_joints]
        return False, "; ".join(dict.fromkeys(reasons))

    def status_line(self) -> str:
        ready = sorted(self.ready_joints())
        ready_text = " ".join(
            f"J{j}={self.trusted[j]}[{self.local_limits[j][0]}..{self.local_limits[j][1]}]"
            for j in ready
        ) or "none"
        blocked = [j for j in self.cfg.allowed_joints if j not in ready]
        blocked_text = " ".join(f"J{j}({self.reasons.get(j, 'blocked')})" for j in blocked)
        return f"ready: {ready_text}" + (f" | blocked: {blocked_text}" if blocked_text else "")

    def validate(self, joint: int, raw_target: float, runtime_ms: int, now: float | None = None) -> SafetyResult:
        ready, reason = self.joint_ready(joint, now)
        if not ready:
            return SafetyResult(False, reason)
        if not math.isfinite(raw_target):
            return SafetyResult(False, "target is not finite")

        current = self.trusted[joint]
        target = int(round(raw_target))
        hard_lo, hard_hi = self.cfg.hard_limits[joint]
        if not hard_lo <= target <= hard_hi:
            return SafetyResult(False, f"target {target} outside firmware range {hard_lo}..{hard_hi}")

        local_lo, local_hi = self.local_limits[joint]
        if not local_lo <= target <= local_hi:
            return SafetyResult(False, f"target {target} outside startup safety envelope {local_lo}..{local_hi}")

        delta = abs(target - current)
        if delta > self.cfg.max_step_deg:
            return SafetyResult(False, f"step {delta} exceeds {self.cfg.max_step_deg}")
        seconds = runtime_ms / 1000.0
        if seconds <= 0:
            return SafetyResult(False, "invalid runtime")
        if delta / seconds > self.cfg.max_speed_deg_s + 1e-9:
            return SafetyResult(False, "speed limit exceeded")
        return SafetyResult(True, target=target)

    def note_command(self, joint: int, target: int, stamp: float | None = None) -> None:
        now = time.monotonic() if stamp is None else stamp
        if joint in self.cfg.allowed_joints:
            self.trusted[joint] = int(target)
            self.last_command_stamp[joint] = now
