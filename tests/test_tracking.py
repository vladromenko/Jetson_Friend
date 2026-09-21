import unittest
from types import SimpleNamespace

from milo.tracking import FaceFollower


def cfg():
    return SimpleNamespace(
        target_x=0.5, target_y=0.5, deadband_x=0.055, deadband_y=0.065,
        max_step_deg=6, face_gain_x=20, face_gain_y=20, pan_joint=1, pan_sign=1,
        tilt_joint_a=3, tilt_sign_a=1, tilt_joint_b=4, tilt_sign_b=1,
        roll_joint=5, roll_sign=1, roll_deadband_deg=12,
    )


class TrackingTests(unittest.TestCase):
    def test_pan_direction_from_real_run(self):
        f=FaceFollower(cfg())
        joint,target=f.plan(0.75,0.5,0,{1:104},{1})
        self.assertEqual(joint,1)
        self.assertGreater(target,104)

    def test_vertical_prefers_j4(self):
        f=FaceFollower(cfg())
        joint,target=f.plan(0.5,0.15,0,{3:90,4:42},{3,4})
        self.assertEqual(joint,4)
        self.assertGreater(target,42)

    def test_vertical_falls_back_to_j3(self):
        f=FaceFollower(cfg())
        joint,target=f.plan(0.5,0.15,0,{3:90},{3})
        self.assertEqual(joint,3)
        self.assertGreater(target,90)

    def test_vertical_uses_j3_when_j4_is_at_lower_stop(self):
        f = FaceFollower(cfg())
        joint, target = f.plan(0.5, 0.72, 0, {3: 96, 4: 0}, {3, 4}, {3: (71, 121), 4: (0, 35)})
        self.assertEqual(joint, 3)
        self.assertLess(target, 96)

    def test_vertical_uses_remaining_j4_travel(self):
        f = FaceFollower(cfg())
        joint, target = f.plan(0.5, 0.15, 0, {3: 96, 4: 34}, {3, 4}, {3: (71, 121), 4: (0, 35)})
        self.assertEqual((joint, target), (4, 35))

    def test_pan_at_limit_does_not_starve_vertical_axis(self):
        f = FaceFollower(cfg())
        joint, target = f.plan(0.8, 0.2, 0, {1: 114, 4: 5}, {1, 4}, {1: (24, 114), 4: (0, 35)})
        self.assertEqual(joint, 4)
        self.assertGreater(target, 5)

    def test_no_auto_reverse(self):
        f=FaceFollower(cfg())
        self.assertEqual(f.pan_sign,1)

    def test_step_grows_with_error_but_stays_bounded(self):
        f = FaceFollower(cfg())
        self.assertEqual(f._step(0.04, 0.045, 6, 20), 0)
        self.assertEqual(f._step(0.08, 0.045, 6, 20), 1)
        self.assertEqual(f._step(0.20, 0.045, 6, 20), 4)
        self.assertEqual(f._step(0.40, 0.045, 6, 20), 6)

    def test_only_ready_axes(self):
        f=FaceFollower(cfg())
        self.assertIsNone(f.plan(0.8,0.2,20,{1:150,4:42},{5}))

if __name__ == '__main__':
    unittest.main()
