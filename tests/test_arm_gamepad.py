import time
import unittest
from dataclasses import replace

from arm_gamepad import (
    JOINTS,
    LIMITS,
    MANUAL_MAX_SPEED_DEG_S,
    MANUAL_MAX_STEP_DEG,
    MANUAL_RUNTIME_MS,
    TELEOP_MAX_COMMAND_STEP_DEG,
    ManualSafetyGate,
    SmoothTeleop,
    SpeedControl,
    requested_moves,
    requested_velocities,
)
from milo.config import Config


class Pad:
    axes = [0.8, -0.8, -0.8, 0.8, -1.0, -1.0]
    buttons = {7, 9}

    def get_numaxes(self):
        return len(self.axes)

    def get_axis(self, index):
        return self.axes[index]

    def get_numbuttons(self):
        return 15

    def get_button(self, index):
        return index in self.buttons


class GamepadTests(unittest.TestCase):
    def test_j6_is_not_mapped(self):
        self.assertEqual(requested_moves(Pad()),
                         {1: -1, 2: 1, 3: -1, 4: -1, 5: 1})

    def test_sticks_are_analog_velocity_requests(self):
        velocities = requested_velocities(Pad())
        self.assertLess(velocities[1], 0.0)
        self.assertGreater(velocities[2], 0.0)
        self.assertLess(abs(velocities[1]), 1.0)
        self.assertLess(abs(velocities[2]), 1.0)

    def test_speed_changes_by_five_and_keeps_small_steps(self):
        speed = SpeedControl(base_speed=15, maximum=25)
        self.assertEqual((speed.speed, speed.command_step(), speed.command_runtime_ms()), (15, 3, 180))
        speed.adjust(5)
        self.assertEqual((speed.speed, speed.command_step()), (20, 4))
        speed.adjust(5)
        self.assertEqual((speed.speed, speed.command_step()), (25, 4))
        speed.adjust(-5)
        self.assertEqual((speed.speed, speed.command_step()), (20, 4))

    def test_smooth_teleop_limits_acceleration_and_step_size(self):
        speed = SpeedControl(base_speed=35)
        teleop = SmoothTeleop(speed, now=10.0)
        first = teleop.command_deltas({1: 1.0}, now=10.18)
        second = teleop.command_deltas({1: 1.0}, now=10.36)
        self.assertLessEqual(abs(first.get(1, 0)), TELEOP_MAX_COMMAND_STEP_DEG)
        self.assertLessEqual(abs(second.get(1, 0)), TELEOP_MAX_COMMAND_STEP_DEG)
        self.assertGreaterEqual(abs(second.get(1, 0)), abs(first.get(1, 0)))

    def test_manual_j1_accepts_position_left_by_face_following(self):
        cfg = Config.load(__import__("pathlib").Path("."))
        cfg = replace(cfg, allowed_joints=JOINTS, hard_limits=LIMITS,
                      local_spans={j: 270 for j in JOINTS},
                      runtime_ms=MANUAL_RUNTIME_MS,
                      max_step_deg=MANUAL_MAX_STEP_DEG,
                      max_speed_deg_s=MANUAL_MAX_SPEED_DEG_S)
        gate = ManualSafetyGate(cfg)
        now = time.monotonic()
        feedback = {1: 155, 2: 89, 3: 90, 4: 5, 5: 89, 6: 30}
        for i in range(3):
            gate.update_measured(feedback, now + i * 0.01)
        self.assertTrue(gate.joint_ready(1, now + 0.05)[0])
        self.assertTrue(gate.validate(1, 151, MANUAL_RUNTIME_MS, now + 0.05).ok)

    def test_j6_locked_and_j2_floor(self):
        cfg = Config.load(__import__("pathlib").Path("."))
        cfg = replace(cfg, allowed_joints=JOINTS, hard_limits=LIMITS,
                      local_spans={j: 270 for j in JOINTS})
        gate = ManualSafetyGate(cfg)
        now = time.monotonic()
        for i in range(3):
            gate.update_measured({1: 90, 2: 90, 3: 90, 4: 60, 5: 90, 6: 30}, now + i * 0.01)
        self.assertFalse(gate.validate(6, 30, 2500, now + 0.05).ok)
        self.assertFalse(gate.validate(2, 116, 700, now + 0.05).ok)
        self.assertTrue(gate.validate(2, 91, 700, now + 0.05).ok)


if __name__ == "__main__":
    unittest.main()
