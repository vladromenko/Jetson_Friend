from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FaceObservation:
    center: tuple[float, float] | None = None
    roll_deg: float = 0.0
    score: float = 0.0
    box: tuple[float, float, float, float] | None = None


class _TensorRTEngine:
    def __init__(self, engine_path):
        import numpy as np
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.np = np
        self.trt = trt
        self.cudart = cudart
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.inputs = {}
        self.outputs = {}
        self.stream = None
        self.closed = False

        with open(engine_path, "rb") as handle:
            engine_data = handle.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Could not create TensorRT context: {engine_path}")
        error, stream = cudart.cudaStreamCreate()
        self._check_cuda(error, "cudaStreamCreate")
        self.stream = stream

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Dynamic TensorRT shape is unsupported: {name} {shape}")
            host = np.empty(shape, dtype=dtype)
            error, device = cudart.cudaMalloc(host.nbytes)
            self._check_cuda(error, f"cudaMalloc {name}")
            self.context.set_tensor_address(name, int(device))
            entry = {"host": host, "device": device}
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = entry
            else:
                self.outputs[name] = entry

    def _check_cuda(self, error, operation):
        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed: {error}")

    def infer(self, input_name, input_data):
        np = self.np
        entry = self.inputs[input_name]
        if entry["host"].shape != input_data.shape:
            raise RuntimeError(f"Unexpected TensorRT input shape {input_data.shape}; expected {entry['host'].shape}")
        np.copyto(entry["host"], input_data)
        error = self.cudart.cudaMemcpyAsync(
            int(entry["device"]), entry["host"].ctypes.data, entry["host"].nbytes,
            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        )[0]
        self._check_cuda(error, f"H2D {input_name}")
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        for name, output in self.outputs.items():
            error = self.cudart.cudaMemcpyAsync(
                output["host"].ctypes.data, int(output["device"]), output["host"].nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
            )[0]
            self._check_cuda(error, f"D2H {name}")
        error = self.cudart.cudaStreamSynchronize(self.stream)[0]
        self._check_cuda(error, "cudaStreamSynchronize")
        return {name: output["host"].copy() for name, output in self.outputs.items()}

    def close(self):
        if self.closed:
            return
        self.closed = True
        for entry in list(self.inputs.values()) + list(self.outputs.values()):
            try:
                self.cudart.cudaFree(entry["device"])
            except Exception:
                pass
        if self.stream is not None:
            try:
                self.cudart.cudaStreamDestroy(self.stream)
            except Exception:
                pass


class _DnnEngine:
    def __init__(self, path):
        import cv2
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.names = self.net.getUnconnectedOutLayersNames()

    def infer(self, input_name, data):
        self.net.setInput(data, input_name)
        return dict(zip(self.names, self.net.forward(self.names)))

    def close(self):
        self.net = None


class _YuNetDecoder:
    INPUT_SIZE = 640

    def __init__(self, engine, score_threshold):
        self.engine = engine
        self.strides = [8, 16, 32]
        self.score_threshold = float(score_threshold)
        self.nms_threshold = 0.30

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return 0.0 if union <= 0.0 else intersection / union

    def _nms(self, detections):
        selected = []
        for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
            keep = all(self._iou(detection["box"], chosen["box"]) < self.nms_threshold for chosen in selected)
            if keep:
                selected.append(detection)
        return selected[:20]

    def infer(self, frame):
        import cv2
        import numpy as np

        image = cv2.resize(frame, (self.INPUT_SIZE, self.INPUT_SIZE)).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image, dtype=np.float32)
        outputs = self.engine.infer("input", image)
        detections = []

        for stride in self.strides:
            cls = outputs[f"cls_{stride}"][0, :, 0]
            obj = outputs[f"obj_{stride}"][0, :, 0]
            bbox = outputs[f"bbox_{stride}"][0]
            kps_output = outputs.get(f"kps_{stride}")
            kps = kps_output[0] if kps_output is not None else None
            scores = np.sqrt(np.clip(cls * obj, 0.0, 1.0))
            indexes = np.where(scores >= self.score_threshold)[0]
            feature_width = self.INPUT_SIZE // stride
            for index in indexes:
                grid_x = index % feature_width
                grid_y = index // feature_width
                anchor_x = grid_x * stride
                anchor_y = grid_y * stride
                center_x = float(bbox[index, 0]) * stride + anchor_x
                center_y = float(bbox[index, 1]) * stride + anchor_y
                width = math.exp(float(bbox[index, 2])) * stride
                height = math.exp(float(bbox[index, 3])) * stride
                landmarks = []
                if kps is not None and len(kps.shape) >= 2 and kps.shape[1] >= 10:
                    for point_index in range(5):
                        point_x = float(kps[index, point_index * 2]) * stride + anchor_x
                        point_y = float(kps[index, point_index * 2 + 1]) * stride + anchor_y
                        landmarks.append((point_x / self.INPUT_SIZE, point_y / self.INPUT_SIZE))
                detections.append({
                    "box": (
                        (center_x - width / 2) / self.INPUT_SIZE,
                        (center_y - height / 2) / self.INPUT_SIZE,
                        (center_x + width / 2) / self.INPUT_SIZE,
                        (center_y + height / 2) / self.INPUT_SIZE,
                    ),
                    "score": float(scores[index]),
                    "landmarks": landmarks,
                })

        result = []
        for detection in self._nms(detections):
            x1, y1, x2, y2 = detection["box"]
            x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in (x1, y1, x2, y2)]
            result.append({
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "box": (x1, y1, x2, y2),
                "confidence": detection["score"],
                "landmarks": detection["landmarks"],
            })
        return result

    def close(self):
        try:
            self.engine.close()
        except Exception:
            pass


class _HaarDetector:
    def __init__(self):
        import cv2
        from pathlib import Path

        candidates = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        ]
        data = getattr(cv2, "data", None)
        if data is not None and getattr(data, "haarcascades", None):
            candidates.insert(0, data.haarcascades + "haarcascade_frontalface_default.xml")
        path = next((candidate for candidate in candidates if Path(candidate).exists()), None)
        if path is None:
            raise RuntimeError("OpenCV Haar cascade not found")
        self.detector = cv2.CascadeClassifier(path)
        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade: {path}")
        self.path = path

    def infer(self, frame):
        import cv2
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=4, minSize=(35, 35))
        result = []
        for x, y, fw, fh in faces:
            if 35 <= fw <= 0.85 * w and 35 <= fh <= 0.85 * h:
                result.append({
                    "center": ((x + fw / 2.0) / w, (y + fh / 2.0) / h),
                    "box": (x / w, y / h, (x + fw) / w, (y + fh) / h),
                    "confidence": 0.5,
                    "landmarks": [],
                })
        return result

    def close(self):
        return None


class VisionEngine:
    def __init__(self, cfg):
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        self.cfg = cfg
        self.detector = None
        self.backend = "none"
        self._smooth_center = None
        self._confirm_count = 0
        self._load_detector()

    def _load_detector(self):
        errors = []
        if self.cfg.face_engine.exists():
            try:
                decoder = _YuNetDecoder(_TensorRTEngine(self.cfg.face_engine), self.cfg.face_score)
                decoder.infer(self.np.zeros((480, 640, 3), dtype=self.np.uint8))
                self.detector = decoder
                self.backend = "TensorRT YuNet"
                print(f"[VISION] face detector={self.backend}", flush=True)
                return
            except Exception as exc:
                errors.append(f"TensorRT YuNet: {exc}")

        if self.cfg.face_model.exists():
            try:
                decoder = _YuNetDecoder(_DnnEngine(self.cfg.face_model), self.cfg.face_score)
                decoder.infer(self.np.zeros((480, 640, 3), dtype=self.np.uint8))
                self.detector = decoder
                self.backend = "OpenCV DNN YuNet compatibility decoder"
                print(f"[VISION] face detector={self.backend}", flush=True)
                return
            except Exception as exc:
                errors.append(f"OpenCV DNN YuNet: {exc}")

        try:
            self.detector = _HaarDetector()
            self.backend = "OpenCV Haar fallback"
            print(f"[VISION] face detector={self.backend}", flush=True)
            if errors:
                print("[VISION] YuNet backends unavailable: " + " | ".join(errors), flush=True)
        except Exception as exc:
            errors.append(f"Haar: {exc}")
            raise RuntimeError("No usable face detector: " + " | ".join(errors)) from exc

    def _switch_to_haar(self, failure):
        if self.backend == "OpenCV Haar fallback":
            raise failure
        try:
            if self.detector is not None:
                self.detector.close()
        except Exception:
            pass
        self.detector = _HaarDetector()
        print(f"[VISION] runtime detector failure ({failure}); switched to Haar fallback", flush=True)
        self.backend = "OpenCV Haar fallback"

    def detect_face(self, frame) -> FaceObservation:
        if frame is None:
            self._confirm_count = 0
            self._smooth_center = None
            return FaceObservation()
        try:
            detections = self.detector.infer(frame)
        except Exception as exc:
            self._switch_to_haar(exc)
            detections = self.detector.infer(frame)
        if not detections:
            self._confirm_count = 0
            self._smooth_center = None
            return FaceObservation()

        face = max(detections, key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]))
        center = face["center"]
        alpha = max(0.0, min(1.0, self.cfg.face_smooth_alpha))
        if self._smooth_center is None:
            self._smooth_center = center
        else:
            self._smooth_center = (
                alpha * center[0] + (1.0 - alpha) * self._smooth_center[0],
                alpha * center[1] + (1.0 - alpha) * self._smooth_center[1],
            )
        self._confirm_count += 1
        if self._confirm_count < self.cfg.face_confirm_frames:
            return FaceObservation()

        roll = 0.0
        landmarks = face.get("landmarks") or []
        if len(landmarks) >= 2:
            ex1, ey1 = landmarks[0]
            ex2, ey2 = landmarks[1]
            roll = math.degrees(math.atan2(ey2 - ey1, ex2 - ex1))
        return FaceObservation(self._smooth_center, roll, float(face.get("confidence", 0.0)), face["box"])

    def jpeg(self, frame, quality=75) -> bytes | None:
        if frame is None:
            return None
        ok, encoded = self.cv2.imencode(".jpg", frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), quality])
        return encoded.tobytes() if ok else None

    def close(self):
        try:
            if self.detector is not None:
                self.detector.close()
        except Exception:
            pass
