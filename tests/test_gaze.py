import unittest

from milo.gaze import camera_to_gaze


class GazeTests(unittest.TestCase):
    def test_cat_eyes_mirror_camera_horizontal_axis(self):
        x, y = camera_to_gaze(0.8, 0.3, -1, 1)
        self.assertAlmostEqual(x, -0.6)
        self.assertAlmostEqual(y, -0.4)

    def test_gaze_is_bounded(self):
        self.assertEqual(camera_to_gaze(2.0, -1.0, -1, 1), (-1.0, -1.0))


if __name__ == "__main__":
    unittest.main()
