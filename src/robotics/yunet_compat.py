"""Reuse MILO's YuNet decoder with OpenCV DNN when FaceDetectorYN is incompatible."""
import cv2
import numpy as np
from vision import TensorRTYuNet


class DnnEngine:
    def __init__(self, path):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.names = self.net.getUnconnectedOutLayersNames()

    def infer(self, input_name, data):
        self.net.setInput(data, input_name)
        return dict(zip(self.names, self.net.forward(self.names)))

    def close(self):
        self.net = None


class YuNetCompat(TensorRTYuNet):
    def __init__(self, path, score):
        self.engine = DnnEngine(path)
        self.strides = [8,16,32]
        self.score_threshold = score
        self.nms_threshold = 0.3
        self.infer(np.zeros((480,640,3), dtype=np.uint8))
