from __future__ import annotations

import math


class FaceFollower:
    """Stable closed-loop face follower.

    The fixed J1 direction follows the measured RC9 run: decreasing J1 moved
    the face farther right in the image. J1 handles horizontal pan. The active
    profile uses J3 for vertical tracking while J4 remains locked.
    J5 is deliberately not moved for ordinary face centering.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.pan_sign = 1 if cfg.pan_sign >= 0 else -1
        self.tilt_signs = {
            cfg.tilt_joint_a: 1 if cfg.tilt_sign_a >= 0 else -1,
            cfg.tilt_joint_b: 1 if cfg.tilt_sign_b >= 0 else -1,
        }
        self.axis_turn = 0

    @staticmethod
    def _step(error: float, deadband: float, maximum: int, gain: float) -> int:
        mag = abs(error)
        if mag <= deadband:
            return 0
        return min(maximum, max(1, math.ceil((mag - deadband) * gain)))

    def motion_hold_reason(self, box):
        if box is None:
            return "face bounds unavailable"
        x1, y1, x2, y2 = box
        if x2 - x1 >= self.cfg.face_max_width or y2 - y1 >= self.cfg.face_max_height:
            return "face too close for reliable tracking"
        return ""

    @staticmethod
    def _bounded_target(joint, current, delta, limits):
        target = current + delta
        if limits and joint in limits:
            lo, hi = limits[joint]
            target = min(hi, max(lo, target))
        return target if target != current else None

    def plan(self, cx, cy, roll_deg, positions, ready_joints=None, limits=None):
        ready = set(positions) if ready_joints is None else set(ready_joints)
        ex = cx - self.cfg.target_x
        ey = cy - self.cfg.target_y
        pan = None
        tilt = None

        jpan = self.cfg.pan_joint
        if jpan in ready and jpan in positions:
            step = self._step(ex, self.cfg.deadband_x, self.cfg.max_step_deg, getattr(self.cfg, "face_gain_x", 20.0))
            if step:
                image_direction = 1 if ex > 0 else -1
                target = self._bounded_target(jpan, positions[jpan], self.pan_sign * image_direction * step, limits)
                if target is not None:
                    pan = (jpan, target)

        if abs(ey) > self.cfg.deadband_y:
            # J4 is primary; J3 takes over when J4 has no travel left in the
            # requested direction.
            order = [self.cfg.tilt_joint_b, self.cfg.tilt_joint_a]
            for joint in order:
                if joint in ready and joint in positions:
                    step = self._step(ey, self.cfg.deadband_y, self.cfg.max_step_deg, getattr(self.cfg, "face_gain_y", 20.0))
                    if step:
                        direction = 1 if ey < 0 else -1
                        sign = self.tilt_signs.get(joint, 1)
                        target = self._bounded_target(joint, positions[joint], sign * direction * step, limits)
                        if target is not None:
                            tilt = (joint, target)
                            break

        if pan is None and tilt is None:
            return None
        if pan is not None and tilt is not None:
            # Alternate axes so both coordinates converge without issuing
            # overlapping arm commands.
            chosen = pan if self.axis_turn % 2 == 0 else tilt
            self.axis_turn += 1
            return chosen
        return pan if pan is not None else tilt
