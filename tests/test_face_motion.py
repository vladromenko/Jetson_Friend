import unittest
from types import SimpleNamespace

from milo.tracking import FaceFollower
from milo.vision import VisionEngine


class FakeDetector:
    def __init__(self, results):
        self.results = iter(results)

    def infer(self, frame):
        return next(self.results)


def face(center, box):
    return {"center": center, "box": box, "confidence": 0.9, "landmarks": []}


class FaceMotionTests(unittest.TestCase):
    def test_lost_face_clears_stale_smoothed_position(self):
        vision = VisionEngine.__new__(VisionEngine)
        vision.cfg = SimpleNamespace(face_smooth_alpha=0.35, face_confirm_frames=1)
        vision.detector = FakeDetector([
            [face((0.2, 0.5), (0.1, 0.4, 0.3, 0.6))],
            [],
            [face((0.8, 0.5), (0.7, 0.4, 0.9, 0.6))],
        ])
        vision._smooth_center = None
        vision._confirm_count = 0
        self.assertEqual(vision.detect_face(object()).center, (0.2, 0.5))
        self.assertIsNone(vision.detect_face(object()).center)
        self.assertEqual(vision.detect_face(object()).center, (0.8, 0.5))

    def test_edge_face_can_recenter_but_too_close_holds(self):
        follower = FaceFollower(SimpleNamespace(
            face_max_width=0.48, face_max_height=0.68,
            pan_sign=1, tilt_joint_a=3, tilt_sign_a=1, tilt_joint_b=4, tilt_sign_b=1,
        ))
        self.assertEqual(follower.motion_hold_reason((0.8, 0.2, 1.0, 0.5)), "")
        self.assertEqual(follower.motion_hold_reason((0.2, 0.2, 0.8, 0.7)), "face too close for reliable tracking")
        self.assertEqual(follower.motion_hold_reason((0.2, 0.2, 0.4, 0.4)), "")


if __name__ == "__main__":
    unittest.main()
