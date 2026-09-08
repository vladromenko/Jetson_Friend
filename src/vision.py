import os
import time

import cv2
import numpy as np
import yaml


class Vision:
    def __init__(self):
        self.camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self.face_model = os.getenv("FACE_MODEL", "/app/models/vision/face_detection_yunet.onnx")
        self.object_model = os.getenv("OBJECT_MODEL", "/app/models/vision/yolov8n.onnx")
        self.labels = self._labels(os.getenv("COCO_LABELS", "/app/models/vision/coco.yaml"))
        self.scene = {"faces": 0, "objects": [], "selected_face": None, "updated": 0}
        self.gaze = (0.0, 0.0)
        self.running = False

    def _labels(self, path):
        try:
            data = yaml.safe_load(open(path, "r", encoding="utf-8"))
            names = data.get("names", {})
            return names if isinstance(names, list) else [names[i] for i in sorted(names)]
        except Exception:
            return []

    def run(self, face):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return
        detector = cv2.FaceDetectorYN.create(self.face_model, "", (320, 240), 0.65, 0.3, 5000) if hasattr(cv2, "FaceDetectorYN") else None
        net = cv2.dnn.readNetFromONNX(self.object_model)
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except Exception:
            pass
        self.running = True
        last_obj = 0.0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            small = cv2.resize(frame, (320, 240))
            face_items = []
            if detector is not None:
                detector.setInputSize((320, 240))
                _, faces = detector.detect(small)
            else:
                faces = None
            if faces is not None:
                for row in faces:
                    x, y, fw, fh, conf = row[:5]
                    cx, cy = (x + fw / 2) / 320, (y + fh / 2) / 240
                    face_items.append({"confidence": round(float(conf), 2), "center": [round(cx, 3), round(cy, 3)]})
            selected = max(face_items, key=lambda f: f["confidence"], default=None)
            if selected:
                gx = (selected["center"][0] - 0.5) * 2
                gy = (selected["center"][1] - 0.5) * 2
                self.gaze = (self.gaze[0] * 0.85 + gx * 0.15, self.gaze[1] * 0.85 + gy * 0.15)
                face.set_gaze(*self.gaze)
            elif time.time() - self.scene.get("updated", 0) > 1.5:
                self.gaze = (self.gaze[0] * 0.92, self.gaze[1] * 0.92)
                face.set_gaze(*self.gaze)
            objects = self.scene["objects"]
            if time.time() - last_obj > 1.0:
                objects = self._detect_objects(net, frame)
                last_obj = time.time()
            self.scene = {"faces": len(face_items), "objects": objects, "selected_face": selected, "updated": time.time()}
        cap.release()

    def _detect_objects(self, net, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        out = net.forward()
        pred = np.squeeze(out).T
        found = {}
        for row in pred:
            scores = row[4:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf > 0.45 and cls < len(self.labels):
                found[self.labels[cls]] = max(found.get(self.labels[cls], 0), conf)
        return [k for k, _ in sorted(found.items(), key=lambda x: x[1], reverse=True)[:8]]
