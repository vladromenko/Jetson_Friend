import os
import unittest
from pathlib import Path

from milo.config import Config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(os.environ)
        os.environ['ARM_ALLOWED_JOINTS'] = '1,3,4,5'
        os.environ['J1_HARD_MIN'] = '89'
        os.environ['J1_HARD_MAX'] = '179'
        os.environ['J3_HARD_MIN'] = '75'
        os.environ['J3_HARD_MAX'] = '135'
        os.environ['J4_HARD_MIN'] = '-30'
        os.environ['J4_HARD_MAX'] = '115'
        os.environ['J5_HARD_MIN'] = '70'
        os.environ['J5_HARD_MAX'] = '110'

    def tearDown(self):
        os.environ.clear(); os.environ.update(self.old)

    def test_allowed_joints_exact(self):
        cfg = Config.load(Path('.'))
        self.assertEqual(cfg.allowed_joints, (1,3,4,5))
        self.assertEqual(cfg.pan_sign, 1)
        self.assertEqual(cfg.cat_gaze_x_sign, -1)
        self.assertEqual(cfg.cat_gaze_y_sign, 1)
        self.assertTrue(cfg.startup_lift_enabled)
        self.assertEqual(cfg.startup_j2_target, 115)
        self.assertEqual(cfg.startup_j3_target, 75)
        self.assertEqual(cfg.startup_j4_target, 0)
        self.assertGreaterEqual(cfg.startup_j4_runtime_ms, 20000)

    def test_forbidden_joint_config_rejected(self):
        os.environ['ARM_ALLOWED_JOINTS'] = '1,2,3,4,5'
        with self.assertRaises(ValueError):
            Config.load(Path('.'))

    def test_tracking_ranges_reject_bad_j3_feedback(self):
        cfg = Config.load(Path('.'))
        self.assertEqual(cfg.hard_limits[1], (89,179))
        self.assertEqual(cfg.hard_limits[3], (75,135))
        self.assertEqual(cfg.hard_limits[4], (-30,115))
        self.assertEqual(cfg.hard_limits[5], (70,110))
        self.assertFalse(cfg.hard_limits[3][0] <= -60 <= cfg.hard_limits[3][1])

    def test_startup_scan_is_removed(self):
        root = Path(__file__).resolve().parents[1]
        main_text = (root / "milo" / "main.py").read_text()
        env_text = (root / "config.env.example").read_text()
        self.assertNotIn("_scan_environment", main_text)
        self.assertNotIn("ARM_STARTUP_POSE=", env_text)
        self.assertNotIn("ENVIRONMENT_SCAN=", env_text)

    def test_verified_tracking_envelope_is_fixed_across_restarts(self):
        root = Path(__file__).resolve().parents[1]
        settings = dict(
            line.split("=", 1)
            for line in (root / "config.env.example").read_text().splitlines()
            if line and not line.startswith("#") and "=" in line
        )
        for joint, bounds in {1: (89, 179), 3: (75, 135), 4: (-30, 115), 5: (70, 110)}.items():
            self.assertEqual((int(settings[f"J{joint}_HARD_MIN"]), int(settings[f"J{joint}_HARD_MAX"])), bounds)
            self.assertGreaterEqual(int(settings[f"J{joint}_LOCAL_SPAN"]), bounds[1] - bounds[0])


if __name__ == '__main__': unittest.main()


def test_raw_topic_present():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "config.env.example").read_text()
    assert "ARM_RAW_TOPIC=/arm6_raw" in text
