import os
import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from milo.config import Config
from milo.safety import ArmSafetyGate
from milo.ros_runtime import RosRuntime


class FakePublisher:
    def __init__(self):
        self.messages = []

    def get_subscription_count(self):
        return 1

    def publish(self, msg):
        self.messages.append(msg)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(os.environ)
        os.environ['ARM_ALLOWED_JOINTS'] = '1,3,4,5'
        os.environ['ARM_FEEDBACK_STABLE_SAMPLES'] = '3'
        self.cfg = Config.load(Path('.'))

    def tearDown(self):
        os.environ.clear(); os.environ.update(self.old)

    def stable_gate(self):
        gate = ArmSafetyGate(self.cfg)
        sample = {1:143,2:188,3:-60,4:33,5:83,6:10}
        for _ in range(3):
            gate.update_measured(sample)
        return gate

    def test_bad_j3_does_not_block_safe_axes(self):
        gate = self.stable_gate()
        self.assertEqual(gate.ready_joints(), {1,4,5})
        self.assertNotIn(3, gate.anchor)
        self.assertIn('outside command range', gate.reasons[3])

    def test_latest_observed_feedback_is_usable(self):
        gate = ArmSafetyGate(self.cfg)
        sample = {1:150,2:89,3:96,4:-3,5:81,6:10}
        for _ in range(3):
            gate.update_measured(sample)
        self.assertEqual(gate.ready_joints(), {1,3,4,5})
        self.assertIn(4, gate.trusted_positions())

    def test_j2_and_j6_always_blocked(self):
        gate = self.stable_gate()
        self.assertFalse(gate.validate(2, 90, 500).ok)
        self.assertFalse(gate.validate(6, 90, 500).ok)

    def test_auxiliary_path_locks_j4_j6_and_j2_box_limit(self):
        runtime = RosRuntime(self.cfg)
        self.assertFalse(runtime.command_aux_joint(4, 0, 500)[0])
        self.assertFalse(runtime.command_aux_joint(6, 27, 500)[0])
        runtime.node = SimpleNamespace(pub=SimpleNamespace(get_subscription_count=lambda: 1))
        runtime.latest_feedback = {2: 115}
        runtime.feedback_stamp = time.monotonic()
        ok, reason = runtime.command_aux_joint(2, 120, 500)
        self.assertFalse(ok)
        self.assertIn("toward the box", reason)

    def test_auxiliary_path_limits_startup_step(self):
        runtime = RosRuntime(self.cfg)
        runtime.node = SimpleNamespace(pub=SimpleNamespace(get_subscription_count=lambda: 1, publish=lambda _msg: None))
        runtime.latest_feedback = {2: 130}
        runtime.feedback_stamp = time.monotonic()
        ok, reason = runtime.command_aux_joint(2, 115, 3000)
        self.assertFalse(ok)
        self.assertIn("aux step", reason)

    def test_startup_j4_only_targets_verified_home_angle(self):
        runtime = RosRuntime(self.cfg)
        pub = FakePublisher()
        runtime.node = SimpleNamespace(pub=pub)
        runtime.latest_feedback = {4: 1}
        runtime.feedback_stamp = time.monotonic()
        ok, reason = runtime.command_startup_j4()
        self.assertTrue(ok, reason)
        self.assertEqual(len(pub.messages), 0)

        runtime.latest_feedback = {4: 70}
        pkg = types.ModuleType("arm_msgs")
        msg_mod = types.ModuleType("arm_msgs.msg")
        msg_mod.ArmJoint = type("ArmJoint", (), {})
        with patch.dict(sys.modules, {"arm_msgs": pkg, "arm_msgs.msg": msg_mod}):
            ok, reason = runtime.command_startup_j4()
        self.assertTrue(ok, reason)
        self.assertEqual(pub.messages[-1].id, 4)
        self.assertEqual(pub.messages[-1].joint, 0)

    def test_one_degree_step_allowed_on_ready_axes(self):
        gate = self.stable_gate()
        self.assertTrue(gate.validate(1, 144, 500).ok)
        self.assertTrue(gate.validate(4, 34, 500).ok)
        self.assertFalse(gate.validate(3, 90, 500).ok)

    def test_trusted_state_advances_after_command(self):
        gate = self.stable_gate()
        result = gate.validate(1, 144, 500)
        self.assertTrue(result.ok)
        gate.note_command(1, result.target)
        self.assertEqual(gate.trusted_positions()[1], 144)
        self.assertTrue(gate.validate(1, 145, 500).ok)

    def test_local_envelope_blocks_drift(self):
        gate = self.stable_gate()
        _, hi = gate.local_limits[1]
        self.assertFalse(gate.validate(1, hi + 1, 500).ok)


if __name__ == '__main__': unittest.main()
