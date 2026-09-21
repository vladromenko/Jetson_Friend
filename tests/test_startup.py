import unittest

from milo.startup import step_toward


class StartupTests(unittest.TestCase):
    def test_j2_moves_up_to_startup_target_from_90(self):
        current = 90
        targets = []
        while True:
            nxt = step_toward(current, 115, 4)
            if nxt is None:
                break
            targets.append(nxt)
            current = nxt
        self.assertIn(115, targets)

    def test_j2_moves_down_to_startup_target_from_folded_pose(self):
        current = 166
        targets = []
        while True:
            nxt = step_toward(current, 115, 4)
            if nxt is None:
                break
            targets.append(nxt)
            current = nxt
        self.assertIn(115, targets)


if __name__ == "__main__":
    unittest.main()
