import unittest
import numpy as np

from milo.ros_runtime import RosRuntime


class DummyCfg:
    allowed_joints=(1,3,4,5)
    feedback_stable_samples=3
    hard_limits={1:(0,180),3:(0,180),4:(0,180),5:(0,270)}
    local_spans={1:20,3:15,4:20,5:20}
    feedback_max_jump_deg=12.0
    feedback_stability_deg=3.0
    feedback_max_age=1.0


class Msg:
    pass


class RosDecodeTests(unittest.TestCase):
    def test_rgb8_decode(self):
        r=RosRuntime(DummyCfg())
        m=Msg(); m.encoding='rgb8'; m.height=1; m.width=2; m.step=6
        m.data=bytes([255,0,0, 0,255,0])
        image=r._image_to_numpy(m)
        self.assertEqual(image.shape,(1,2,3))
        self.assertEqual(image[0,0].tolist(),[0,0,255])

    def test_padded_bgr8_decode(self):
        r=RosRuntime(DummyCfg())
        m=Msg(); m.encoding='bgr8'; m.height=1; m.width=1; m.step=4
        m.data=bytes([1,2,3,99])
        image=r._image_to_numpy(m)
        self.assertEqual(image[0,0].tolist(),[1,2,3])


if __name__ == '__main__': unittest.main()
