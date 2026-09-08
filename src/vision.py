import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import yaml


class TensorRTEngine:
    def __init__(self, engine_path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.trt = trt
        self.cudart = cudart
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.inputs = {}
        self.outputs = {}
        self.stream = None
        self.closed = False

        with open(engine_path, "rb") as file:
            engine_data = file.read()

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
                raise RuntimeError(
                    f"Dynamic TensorRT shape is not supported: {name} {shape}"
                )

            host = np.empty(shape, dtype=dtype)
            error, device = cudart.cudaMalloc(host.nbytes)
            self._check_cuda(error, f"cudaMalloc {name}")

            self.context.set_tensor_address(name, int(device))

            entry = {
                "host": host,
                "device": device,
            }

            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = entry
            else:
                self.outputs[name] = entry

    def _check_cuda(self, error, operation):
        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed: {error}")

    def infer(self, input_name, input_data):
        if self.closed:
            raise RuntimeError("TensorRT engine is closed")

        entry = self.inputs[input_name]

        if entry["host"].shape != input_data.shape:
            raise RuntimeError(
                f"Unexpected shape {input_data.shape}; "
                f"expected {entry['host'].shape}"
            )

        np.copyto(entry["host"], input_data)

        error = self.cudart.cudaMemcpyAsync(
            int(entry["device"]),
            entry["host"].ctypes.data,
            entry["host"].nbytes,
            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )[0]
        self._check_cuda(error, f"H2D {input_name}")

        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT execute_async_v3 failed")

        for name, output in self.outputs.items():
            error = self.cudart.cudaMemcpyAsync(
                output["host"].ctypes.data,
                int(output["device"]),
                output["host"].nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )[0]
            self._check_cuda(error, f"D2H {name}")

        error = self.cudart.cudaStreamSynchronize(self.stream)[0]
        self._check_cuda(error, "cudaStreamSynchronize")

        return {
            name: output["host"].copy()
            for name, output in self.outputs.items()
        }

    def close(self):
        if self.closed:
            return

        self.closed = True

        for entry in self.inputs.values():
            self.cudart.cudaFree(entry["device"])

        for entry in self.outputs.values():
            self.cudart.cudaFree(entry["device"])

        if self.stream is not None:
            self.cudart.cudaStreamDestroy(self.stream)

        self.inputs = {}
        self.outputs = {}


class TensorRTYOLO:
    def __init__(self, engine_path):
        self.engine = TensorRTEngine(engine_path)
        print(f"TensorRT YOLO loaded: {engine_path}", flush=True)

    def infer(self, frame):
        image = cv2.resize(frame, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image, dtype=np.float32)
        return self.engine.infer("images", image)["output0"]

    def close(self):
        self.engine.close()


class TensorRTYuNet:
    def __init__(self, engine_path):
        self.engine = TensorRTEngine(engine_path)
        self.strides = [8, 16, 32]
        self.score_threshold = 0.65
        self.nms_threshold = 0.30
        print(f"TensorRT YuNet loaded: {engine_path}", flush=True)

    def infer(self, frame):
        original_height, original_width = frame.shape[:2]

        image = cv2.resize(frame, (640, 640)).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image, dtype=np.float32)

        outputs = self.engine.infer("input", image)
        detections = []

        for stride in self.strides:
            cls = outputs[f"cls_{stride}"][0, :, 0]
            obj = outputs[f"obj_{stride}"][0, :, 0]
            bbox = outputs[f"bbox_{stride}"][0]
            scores = np.sqrt(np.clip(cls * obj, 0.0, 1.0))
            indexes = np.where(scores >= self.score_threshold)[0]
            feature_width = 640 // stride

            for index in indexes:
                grid_x = index % feature_width
                grid_y = index // feature_width
                anchor_x = grid_x * stride
                anchor_y = grid_y * stride

                center_x = bbox[index, 0] * stride + anchor_x
                center_y = bbox[index, 1] * stride + anchor_y
                width = math.exp(float(bbox[index, 2])) * stride
                height = math.exp(float(bbox[index, 3])) * stride

                detections.append(
                    {
                        "box": [
                            float(center_x - width / 2),
                            float(center_y - height / 2),
                            float(center_x + width / 2),
                            float(center_y + height / 2),
                        ],
                        "score": float(scores[index]),
                    }
                )

        detections = self._nms(detections)
        result = []

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]

            cx = ((x1 + x2) / 2) * original_width / 640.0
            cy = ((y1 + y2) / 2) * original_height / 640.0

            result.append(
                {
                    "confidence": round(detection["score"], 2),
                    "center": [
                        round(cx / original_width, 3),
                        round(cy / original_height, 3),
                    ],
                }
            )

        return result

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _nms(self, detections):
        ordered = sorted(
            detections,
            key=lambda item: item["score"],
            reverse=True,
        )
        selected = []

        for detection in ordered:
            keep = True

            for chosen in selected:
                if self._iou(detection["box"], chosen["box"]) >= self.nms_threshold:
                    keep = False
                    break

            if keep:
                selected.append(detection)

        return selected[:50]

    def close(self):
        self.engine.close()


class HaarFaceDetector:
    def __init__(self):
        candidates = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        ]

        package_data = getattr(cv2, "data", None)

        if package_data is not None:
            candidates.append(
                os.path.join(
                    package_data.haarcascades,
                    "haarcascade_frontalface_default.xml",
                )
            )

        cascade_path = None

        for candidate in candidates:
            if os.path.isfile(candidate):
                cascade_path = candidate
                break

        if cascade_path is None:
            raise RuntimeError(
                "OpenCV Haar cascade not found. "
                "Install opencv-data or set a valid cascade path."
            )

        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

        print(
            f"Vision fallback: OpenCV Haar CPU ({cascade_path})",
            flush=True,
        )

    def infer(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(50, 50),
        )

        height, width = frame.shape[:2]
        result = []

        for x, y, w, h in faces:
            result.append(
                {
                    "confidence": 0.60,
                    "center": [
                        round((x + w / 2) / width, 3),
                        round((y + h / 2) / height, 3),
                    ],
                }
            )

        return result


class Vision:
    def __init__(self):
        root = os.getenv(
            "JETSON_FRIEND_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        self.camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self.object_model = os.getenv(
            "OBJECT_MODEL",
            os.path.join(root, "models", "vision", "yolov8n.engine"),
        )
        self.face_model = os.getenv(
            "FACE_MODEL",
            os.path.join(root, "models", "vision", "face_detection_yunet.engine"),
        )

        labels_path = os.getenv(
            "COCO_LABELS",
            os.path.join(root, "models", "vision", "coco.yaml"),
        )

        self.labels = self._labels(labels_path)
        self.scene = {
            "faces": 0,
            "objects": [],
            "selected_face": None,
            "updated": 0.0,
            "vision_backend": "not started",
        }
        self.gaze = (0.0, 0.0)
        self.running = False
        self.backend = "not started"

    def _labels(self, path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

            names = data.get("names", {})

            if isinstance(names, list):
                return names

            return [names[index] for index in sorted(names)]

        except Exception as exc:
            print(f"Could not load labels: {exc}", flush=True)
            return []

    def _tensorrt_preflight(self):
        code = r"""
import sys
import tensorrt as trt
from cuda.bindings import runtime as cudart

error, count = cudart.cudaGetDeviceCount()
if error != cudart.cudaError_t.cudaSuccess or count < 1:
    raise SystemExit(20)

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)

with open(sys.argv[1], "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

if engine is None:
    raise SystemExit(21)
"""
        try:
            process = subprocess.run(
                [sys.executable, "-c", code, self.object_model],
                text=True,
                capture_output=True,
                timeout=15,
            )

            if process.returncode == 0:
                return True

            details = (
                process.stderr.strip()
                or process.stdout.strip()
                or f"exit code {process.returncode}"
            )
            print(
                "TensorRT/CUDA unavailable; using CPU face tracking. "
                f"Details: {details}",
                flush=True,
            )
            return False

        except Exception as exc:
            print(
                f"TensorRT/CUDA preflight error; using CPU face tracking: {exc}",
                flush=True,
            )
            return False

    def _open_camera(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not cap.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self, face):
        self.running = True
        cap = None
        yolo = None
        face_detector = None
        use_tensorrt = self._tensorrt_preflight()

        if use_tensorrt:
            try:
                yolo = TensorRTYOLO(self.object_model)
                face_detector = TensorRTYuNet(self.face_model)
                self.backend = "TensorRT"
            except Exception as exc:
                print(f"TensorRT initialization failed: {exc}", flush=True)
                yolo = None
                face_detector = None
                use_tensorrt = False

        if not use_tensorrt:
            try:
                face_detector = HaarFaceDetector()
                self.backend = "OpenCV Haar CPU"
            except Exception as exc:
                print(f"CPU face fallback failed: {exc}", flush=True)
                self.backend = "camera only"

        last_objects = 0.0
        last_faces = 0.0
        objects = []
        face_items = []
        camera_error_printed = False

        try:
            while self.running:
                if cap is None:
                    cap = self._open_camera()

                    if cap is None:
                        if not camera_error_printed:
                            print(
                                f"Camera could not be opened: {self.camera_index}. Retrying...",
                                flush=True,
                            )
                            camera_error_printed = True
                        time.sleep(1.0)
                    else:
                        camera_error_printed = False
                        print(f"Camera opened: {self.camera_index}", flush=True)

                if cap is not None:
                    ok, frame = cap.read()

                    if not ok:
                        cap.release()
                        cap = None
                        time.sleep(0.2)
                    else:
                        now = time.time()

                        if (
                            face_detector is not None
                            and now - last_faces >= 0.12
                        ):
                            face_items = face_detector.infer(frame)
                            last_faces = now

                        if (
                            yolo is not None
                            and now - last_objects >= 0.8
                        ):
                            output = yolo.infer(frame)
                            objects = self._detect_objects(output)
                            last_objects = now

                        selected = max(
                            face_items,
                            key=lambda item: item["confidence"],
                            default=None,
                        )

                        self._update_gaze(selected, face)

                        self.scene = {
                            "faces": len(face_items),
                            "objects": objects,
                            "selected_face": selected,
                            "updated": now,
                            "vision_backend": self.backend,
                        }

        except Exception as exc:
            print(f"Vision error: {exc}", flush=True)

        finally:
            self.running = False

            if cap is not None:
                cap.release()

            if yolo is not None:
                try:
                    yolo.close()
                except Exception as exc:
                    print(f"YOLO shutdown warning: {exc}", flush=True)

            if use_tensorrt and face_detector is not None:
                try:
                    face_detector.close()
                except Exception as exc:
                    print(f"YuNet shutdown warning: {exc}", flush=True)

    def _update_gaze(self, selected, face):
        if selected is not None:
            camera_x = (selected["center"][0] - 0.5) * 2.0
            camera_y = (selected["center"][1] - 0.5) * 2.0

            gx = camera_x
            gy = camera_y

            self.gaze = (
                self.gaze[0] * 0.72 + gx * 0.28,
                self.gaze[1] * 0.72 + gy * 0.28,
            )
        else:
            self.gaze = (
                self.gaze[0] * 0.94,
                self.gaze[1] * 0.94,
            )

        face.set_gaze(*self.gaze)

    def _detect_objects(self, output):
        predictions = output[0].T
        found = {}

        for row in predictions:
            scores = row[4:]
            cls = int(np.argmax(scores))
            confidence = float(scores[cls])

            if confidence > 0.45 and cls < len(self.labels):
                label = self.labels[cls]
                previous = found.get(label, 0.0)

                if confidence > previous:
                    found[label] = confidence

        ordered = sorted(
            found.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [label for label, confidence in ordered[:8]]

    def stop(self):
        self.running = False
