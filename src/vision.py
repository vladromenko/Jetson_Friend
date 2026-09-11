import math
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import yaml

from tracking import FaceTracks


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

            host = np.empty(
                shape,
                dtype=dtype,
            )

            error, device = cudart.cudaMalloc(host.nbytes)
            self._check_cuda(
                error,
                f"cudaMalloc {name}",
            )

            self.context.set_tensor_address(
                name,
                int(device),
            )

            entry = {
                "host": host,
                "device": device,
            }

            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = entry
            else:
                self.outputs[name] = entry

    def _check_cuda(
        self,
        error,
        operation,
    ):
        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed: {error}")

    def infer(
        self,
        input_name,
        input_data,
    ):
        if self.closed:
            raise RuntimeError("TensorRT engine is closed")

        entry = self.inputs[input_name]

        if entry["host"].shape != input_data.shape:
            raise RuntimeError(
                f"Unexpected shape {input_data.shape}; expected {entry['host'].shape}"
            )

        np.copyto(
            entry["host"],
            input_data,
        )

        error = self.cudart.cudaMemcpyAsync(
            int(entry["device"]),
            entry["host"].ctypes.data,
            entry["host"].nbytes,
            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )[0]

        self._check_cuda(
            error,
            f"H2D {input_name}",
        )

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

            self._check_cuda(
                error,
                f"D2H {name}",
            )

        error = self.cudart.cudaStreamSynchronize(self.stream)[0]

        self._check_cuda(
            error,
            "cudaStreamSynchronize",
        )

        return {name: output["host"].copy() for name, output in self.outputs.items()}

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
    INPUT_SIZE = 640

    def __init__(
        self,
        engine_path,
    ):
        self.engine = TensorRTEngine(engine_path)

        print(
            f"TensorRT YOLO loaded: {engine_path}",
            flush=True,
        )

    def infer(
        self,
        frame,
    ):
        image = cv2.resize(
            frame,
            (
                self.INPUT_SIZE,
                self.INPUT_SIZE,
            ),
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = image.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        image = np.expand_dims(
            image,
            axis=0,
        )

        image = np.ascontiguousarray(
            image,
            dtype=np.float32,
        )

        return self.engine.infer(
            "images",
            image,
        )["output0"]

    def close(self):
        self.engine.close()


class TensorRTYuNet:
    INPUT_SIZE = 640

    def __init__(
        self,
        engine_path,
    ):
        self.engine = TensorRTEngine(engine_path)

        self.strides = [
            8,
            16,
            32,
        ]

        self.score_threshold = 0.65
        self.nms_threshold = 0.30

        print(
            f"TensorRT YuNet loaded: {engine_path}",
            flush=True,
        )

    def infer(
        self,
        frame,
    ):
        original_height, original_width = frame.shape[:2]

        image = cv2.resize(
            frame,
            (
                self.INPUT_SIZE,
                self.INPUT_SIZE,
            ),
        ).astype(np.float32)

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        image = np.expand_dims(
            image,
            axis=0,
        )

        image = np.ascontiguousarray(
            image,
            dtype=np.float32,
        )

        outputs = self.engine.infer(
            "input",
            image,
        )

        detections = []

        for stride in self.strides:
            cls = outputs[f"cls_{stride}"][0, :, 0]

            obj = outputs[f"obj_{stride}"][0, :, 0]

            bbox = outputs[f"bbox_{stride}"][0]

            kps_output = outputs.get(f"kps_{stride}")

            if kps_output is not None:
                kps = kps_output[0]
            else:
                kps = None

            scores = np.sqrt(
                np.clip(
                    cls * obj,
                    0.0,
                    1.0,
                )
            )

            indexes = np.where(scores >= self.score_threshold)[0]

            feature_width = self.INPUT_SIZE // stride

            for index in indexes:
                grid_x = index % feature_width

                grid_y = index // feature_width

                anchor_x = grid_x * stride

                anchor_y = grid_y * stride

                center_x = bbox[index, 0] * stride + anchor_x

                center_y = bbox[index, 1] * stride + anchor_y

                width = (
                    math.exp(
                        float(
                            bbox[
                                index,
                                2,
                            ]
                        )
                    )
                    * stride
                )

                height = (
                    math.exp(
                        float(
                            bbox[
                                index,
                                3,
                            ]
                        )
                    )
                    * stride
                )

                landmarks = []

                if kps is not None and len(kps.shape) >= 2 and kps.shape[1] >= 10:
                    for point_index in range(5):
                        point_x = (
                            float(
                                kps[
                                    index,
                                    point_index * 2,
                                ]
                            )
                            * stride
                            + anchor_x
                        )

                        point_y = (
                            float(
                                kps[
                                    index,
                                    point_index * 2 + 1,
                                ]
                            )
                            * stride
                            + anchor_y
                        )

                        landmarks.append(
                            [
                                point_x,
                                point_y,
                            ]
                        )

                detections.append(
                    {
                        "box": [
                            float(center_x - width / 2),
                            float(center_y - height / 2),
                            float(center_x + width / 2),
                            float(center_y + height / 2),
                        ],
                        "score": float(scores[index]),
                        "landmarks": landmarks,
                    }
                )

        detections = self._nms(detections)

        result = []

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]

            x1 = max(
                0.0,
                min(
                    float(self.INPUT_SIZE),
                    x1,
                ),
            )

            y1 = max(
                0.0,
                min(
                    float(self.INPUT_SIZE),
                    y1,
                ),
            )

            x2 = max(
                0.0,
                min(
                    float(self.INPUT_SIZE),
                    x2,
                ),
            )

            y2 = max(
                0.0,
                min(
                    float(self.INPUT_SIZE),
                    y2,
                ),
            )

            nx1 = x1 / self.INPUT_SIZE

            ny1 = y1 / self.INPUT_SIZE

            nx2 = x2 / self.INPUT_SIZE

            ny2 = y2 / self.INPUT_SIZE

            cx = (nx1 + nx2) / 2.0

            cy = (ny1 + ny2) / 2.0

            area = max(
                0.0,
                nx2 - nx1,
            ) * max(
                0.0,
                ny2 - ny1,
            )

            normalized_landmarks = []

            for point in detection.get(
                "landmarks",
                [],
            ):
                normalized_landmarks.append(
                    [
                        round(
                            max(
                                0.0,
                                min(
                                    1.0,
                                    float(point[0]) / self.INPUT_SIZE,
                                ),
                            ),
                            4,
                        ),
                        round(
                            max(
                                0.0,
                                min(
                                    1.0,
                                    float(point[1]) / self.INPUT_SIZE,
                                ),
                            ),
                            4,
                        ),
                    ]
                )

            result.append(
                {
                    "confidence": round(
                        detection["score"],
                        3,
                    ),
                    "center": [
                        round(
                            cx,
                            4,
                        ),
                        round(
                            cy,
                            4,
                        ),
                    ],
                    "box": [
                        round(
                            nx1,
                            4,
                        ),
                        round(
                            ny1,
                            4,
                        ),
                        round(
                            nx2,
                            4,
                        ),
                        round(
                            ny2,
                            4,
                        ),
                    ],
                    "area": round(
                        area,
                        5,
                    ),
                    "landmarks": (normalized_landmarks),
                }
            )

        return result

    @staticmethod
    def _iou(
        box_a,
        box_b,
    ):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(
            ax1,
            bx1,
        )

        iy1 = max(
            ay1,
            by1,
        )

        ix2 = min(
            ax2,
            bx2,
        )

        iy2 = min(
            ay2,
            by2,
        )

        intersection = max(
            0.0,
            ix2 - ix1,
        ) * max(
            0.0,
            iy2 - iy1,
        )

        area_a = max(
            0.0,
            ax2 - ax1,
        ) * max(
            0.0,
            ay2 - ay1,
        )

        area_b = max(
            0.0,
            bx2 - bx1,
        ) * max(
            0.0,
            by2 - by1,
        )

        union = area_a + area_b - intersection

        if union <= 0.0:
            return 0.0

        return intersection / union

    def _nms(
        self,
        detections,
    ):
        ordered = sorted(
            detections,
            key=lambda item: item["score"],
            reverse=True,
        )

        selected = []

        for detection in ordered:
            keep = True

            for chosen in selected:
                if (
                    self._iou(
                        detection["box"],
                        chosen["box"],
                    )
                    >= self.nms_threshold
                ):
                    keep = False
                    break

            if keep:
                selected.append(detection)

        return selected[:20]

    def close(self):
        self.engine.close()


class HaarFaceDetector:
    def __init__(self):
        candidates = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        ]

        package_data = getattr(
            cv2,
            "data",
            None,
        )

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
            raise RuntimeError("OpenCV Haar cascade not found")

        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

        print(
            f"Vision fallback: OpenCV Haar CPU ({cascade_path})",
            flush=True,
        )

    def infer(
        self,
        frame,
    ):
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

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
            nx1 = x / width
            ny1 = y / height

            nx2 = (x + w) / width

            ny2 = (y + h) / height

            result.append(
                {
                    "confidence": 0.60,
                    "center": [
                        round(
                            (nx1 + nx2) / 2.0,
                            4,
                        ),
                        round(
                            (ny1 + ny2) / 2.0,
                            4,
                        ),
                    ],
                    "box": [
                        round(
                            nx1,
                            4,
                        ),
                        round(
                            ny1,
                            4,
                        ),
                        round(
                            nx2,
                            4,
                        ),
                        round(
                            ny2,
                            4,
                        ),
                    ],
                    "area": round(
                        max(
                            0.0,
                            nx2 - nx1,
                        )
                        * max(
                            0.0,
                            ny2 - ny1,
                        ),
                        5,
                    ),
                }
            )

        return result


class Vision:
    """
    Adaptive low-overhead vision loop
    for Jetson Orin Nano.

    Camera capture remains continuous,
    but expensive neural inference is
    event-driven and runs slower than
    camera FPS.
    """

    def __init__(self):
        root = os.getenv(
            "JETSON_FRIEND_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        self.camera_index = int(
            os.getenv(
                "CAMERA_INDEX",
                "0",
            )
        )

        self.object_model = os.getenv(
            "OBJECT_MODEL",
            os.path.join(
                root,
                "models",
                "vision",
                "yolov8n.engine",
            ),
        )

        self.face_model = os.getenv(
            "FACE_MODEL",
            os.path.join(
                root,
                "models",
                "vision",
                "face_detection_yunet.engine",
            ),
        )

        labels_path = os.getenv(
            "COCO_LABELS",
            os.path.join(
                root,
                "models",
                "vision",
                "coco.yaml",
            ),
        )

        self.labels = self._labels(labels_path)

        self.face_interval_active = float(
            os.getenv(
                "VISION_FACE_ACTIVE_SEC",
                "0.35",
            )
        )

        self.face_interval_idle = float(
            os.getenv(
                "VISION_FACE_IDLE_SEC",
                "1.20",
            )
        )

        self.object_interval_active = float(
            os.getenv(
                "VISION_OBJECT_ACTIVE_SEC",
                "1.50",
            )
        )

        self.object_interval_idle = float(
            os.getenv(
                "VISION_OBJECT_IDLE_SEC",
                "5.00",
            )
        )

        self.motion_interval = float(
            os.getenv(
                "VISION_MOTION_SEC",
                "0.20",
            )
        )

        self.motion_threshold = float(
            os.getenv(
                "VISION_MOTION_THRESHOLD",
                "0.025",
            )
        )

        self.presence_hold = float(
            os.getenv(
                "VISION_PRESENCE_HOLD_SEC",
                "2.0",
            )
        )

        self.object_hold = float(
            os.getenv(
                "VISION_OBJECT_HOLD_SEC",
                "8.0",
            )
        )

        self.scene_lock = threading.RLock()

        self.frame_lock = threading.RLock()

        self.attention_lock = threading.RLock()

        self.scene = {
            "faces": 0,
            "objects": [],
            "selected_face": None,
            "motion": 0.0,
            "person_present": False,
            "updated": 0.0,
            "vision_backend": ("not started"),
        }

        self.gaze = (
            0.0,
            0.0,
        )

        self.running = False
        self.backend = "not started"

        self.latest_frame = None
        self.stats = {
            "frames": 0,
            "face_calls": 0,
            "object_calls": 0,
            "face_ms": 0.0,
            "object_ms": 0.0,
        }
        self.stats_started = None
        self.face_tracks = FaceTracks()
        self.face_samples = {}
        self.latest_face_crop = None
        self.latest_face_meta = None

        self.visual_attention_until = 0.0
        self.last_person_seen = 0.0
        self.last_object_update = 0.0

        self.previous_motion_frame = None

    def _labels(
        self,
        path,
    ):
        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:
                data = yaml.safe_load(file)

            names = data.get(
                "names",
                {},
            )

            if isinstance(
                names,
                list,
            ):
                return names

            return [names[index] for index in sorted(names)]

        except Exception as exc:
            print(
                f"Could not load labels: {exc}",
                flush=True,
            )

            return []

    def _tensorrt_preflight(
        self,
    ):
        code = r"""
import sys
import tensorrt as trt
from cuda.bindings import runtime as cudart

error, count = cudart.cudaGetDeviceCount()

if (
    error != cudart.cudaError_t.cudaSuccess
    or count < 1
):
    raise SystemExit(20)

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)

with open(sys.argv[1], "rb") as f:
    engine = runtime.deserialize_cuda_engine(
        f.read()
    )

if engine is None:
    raise SystemExit(21)
"""

        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    self.object_model,
                ],
                text=True,
                capture_output=True,
                timeout=15,
            )

            if process.returncode == 0:
                return True

            details = (
                process.stderr.strip()
                or process.stdout.strip()
                or (f"exit code {process.returncode}")
            )

            print(
                "TensorRT/CUDA unavailable; "
                "using CPU face tracking. "
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

    def _open_camera(
        self,
    ):
        cap = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_V4L2,
        )

        if not cap.isOpened():
            cap.release()
            return None

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        return cap

    def request_visual_attention(
        self,
        seconds=3.0,
    ):
        """
        Temporarily increase vision
        frequency for questions such as
        'What can you see?'.
        """

        now = time.monotonic()

        with self.attention_lock:
            self.visual_attention_until = max(
                self.visual_attention_until,
                now
                + max(
                    0.5,
                    float(seconds),
                ),
            )

    def _visual_attention_active(
        self,
    ):
        with self.attention_lock:
            return time.monotonic() < self.visual_attention_until

    def get_scene(
        self,
    ):
        with self.scene_lock:
            return dict(self.scene)

    def run(
        self,
        face,
    ):
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
                print(
                    f"TensorRT initialization failed: {exc}",
                    flush=True,
                )

                if yolo is not None:
                    yolo.close()
                if face_detector is not None:
                    face_detector.close()
                yolo = None
                face_detector = None
                use_tensorrt = False

        if not use_tensorrt:
            try:
                face_detector = HaarFaceDetector()

                self.backend = "OpenCV Haar CPU"

            except Exception as exc:
                print(
                    f"CPU face fallback failed: {exc}",
                    flush=True,
                )

                self.backend = "camera only"

        last_objects = 0.0
        last_faces = 0.0
        last_motion = 0.0

        objects = []
        face_items = []
        motion_score = 0.0

        camera_error_printed = False

        try:
            while self.running:
                while self.running and cap is None:
                    cap = self._open_camera()

                    if cap is None:
                        if not camera_error_printed:
                            print(
                                "Camera could not "
                                "be opened: "
                                f"{self.camera_index}. "
                                "Retrying...",
                                flush=True,
                            )

                            camera_error_printed = True

                        time.sleep(1.0)

                    else:
                        camera_error_printed = False

                        self.previous_motion_frame = None

                        print(
                            f"Camera opened: {self.camera_index}",
                            flush=True,
                        )

                if not self.running:
                    break

                ok, frame = cap.read()

                while self.running and not ok:
                    with self.frame_lock:
                        self.face_samples = {}
                        self.latest_frame = None
                    with self.scene_lock:
                        self.scene = {
                            "faces": 0,
                            "face_tracks": [],
                            "objects": [],
                            "updated": 0.0,
                            "person_present": False,
                        }
                    cap.release()
                    cap = None

                    time.sleep(0.2)

                    while self.running and cap is None:
                        cap = self._open_camera()

                        if cap is None:
                            if not camera_error_printed:
                                print(
                                    "Camera connection lost. Retrying...",
                                    flush=True,
                                )

                                camera_error_printed = True

                            time.sleep(1.0)

                        else:
                            camera_error_printed = False

                            self.previous_motion_frame = None

                            print(
                                f"Camera reopened: {self.camera_index}",
                                flush=True,
                            )

                    if self.running and cap is not None:
                        ok, frame = cap.read()

                if not self.running:
                    break

                now = time.monotonic()
                wall_now = time.time()
                if self.stats_started is None:
                    self.stats_started = now
                self.stats["frames"] += 1

                with self.frame_lock:
                    self.latest_frame = frame.copy()

                if now - last_motion >= self.motion_interval:
                    motion_score = self._motion_score(frame)

                    last_motion = now

                had_recent_person = (
                    wall_now - self.last_person_seen <= self.presence_hold
                )

                attention = self._visual_attention_active()

                activity = motion_score >= self.motion_threshold

                active_scene = had_recent_person or activity or attention

                if active_scene:
                    face_interval = self.face_interval_active

                else:
                    face_interval = self.face_interval_idle

                if face_detector is not None and now - last_faces >= face_interval:
                    try:
                        inference_start = time.monotonic()
                        face_items = self.face_tracks.update(
                            face_detector.infer(frame), now
                        )
                        self.stats["face_calls"] += 1
                        self.stats["face_ms"] = round(
                            (time.monotonic() - inference_start) * 1000, 2
                        )

                    except Exception as exc:
                        print(
                            f"Face inference warning: {exc}",
                            flush=True,
                        )

                        face_items = []

                    last_faces = now

                    if face_items:
                        self.last_person_seen = wall_now

                person_present = bool(face_items) or (
                    wall_now - self.last_person_seen <= self.presence_hold
                )

                if person_present or activity or attention:
                    object_interval = self.object_interval_active

                else:
                    object_interval = self.object_interval_idle

                should_run_objects = (
                    yolo is not None
                    and now - last_objects >= object_interval
                    and (attention or activity or person_present or not objects)
                )

                if should_run_objects:
                    try:
                        inference_start = time.monotonic()
                        output = yolo.infer(frame)
                        self.stats["object_calls"] += 1
                        self.stats["object_ms"] = round(
                            (time.monotonic() - inference_start) * 1000, 2
                        )

                        objects = self._detect_objects(output)

                        self.last_object_update = wall_now

                    except Exception as exc:
                        print(
                            f"YOLO inference warning: {exc}",
                            flush=True,
                        )

                    last_objects = now

                if (
                    objects
                    and not person_present
                    and not activity
                    and (wall_now - self.last_object_update > self.object_hold)
                ):
                    objects = []

                selected = self._select_face(face_items)

                if face_items and face_items[0].get("detected_at") == now:
                    with self.frame_lock:
                        self.face_samples = {}
                    for item in face_items:
                        self._cache_face(frame, item)
                elif not face_items:
                    with self.frame_lock:
                        self.face_samples = {}
                        self.latest_face_crop = None
                        self.latest_face_meta = None

                self._update_gaze(
                    selected,
                    face,
                )

                with self.scene_lock:
                    self.scene = {
                        "faces": len(face_items),
                        "objects": list(objects),
                        "selected_face": selected,
                        "face_tracks": [dict(item) for item in face_items],
                        "motion": round(
                            float(motion_score),
                            4,
                        ),
                        "person_present": (bool(person_present)),
                        "metrics": {
                            **self.stats,
                            "elapsed_sec": now - self.stats_started,
                        },
                        "updated": wall_now,
                        "vision_backend": (self.backend),
                    }

        except Exception as exc:
            print(
                f"Vision error: {exc}",
                flush=True,
            )

        finally:
            self.running = False

            if cap is not None:
                cap.release()

            if yolo is not None:
                try:
                    yolo.close()

                except Exception as exc:
                    print(
                        f"YOLO shutdown warning: {exc}",
                        flush=True,
                    )

            if use_tensorrt and face_detector is not None:
                try:
                    face_detector.close()

                except Exception as exc:
                    print(
                        f"YuNet shutdown warning: {exc}",
                        flush=True,
                    )

    def _motion_score(
        self,
        frame,
    ):
        small = cv2.resize(
            frame,
            (160, 90),
            interpolation=(cv2.INTER_AREA),
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        previous = self.previous_motion_frame

        self.previous_motion_frame = gray

        if previous is None:
            return 0.0

        difference = cv2.absdiff(
            previous,
            gray,
        )

        changed = np.count_nonzero(difference > 18)

        return changed / float(difference.size)

    @staticmethod
    def _select_face(
        face_items,
    ):
        if not face_items:
            return None

        def score(item):
            confidence = float(
                item.get(
                    "confidence",
                    0.0,
                )
            )

            area = float(
                item.get(
                    "area",
                    0.0,
                )
            )

            center = item.get(
                "center",
                [
                    0.5,
                    0.5,
                ],
            )

            dx = float(center[0]) - 0.5

            dy = float(center[1]) - 0.5

            center_bonus = max(
                0.0,
                1.0 - math.sqrt(dx * dx + dy * dy),
            )

            return (
                confidence * 2.0
                + min(
                    area * 8.0,
                    1.0,
                )
                + center_bonus * 0.25
            )

        return max(
            face_items,
            key=score,
        )

    def _cache_face(self, frame, selected):
        if selected is None:
            return
        box = selected.get("box")
        if not box or len(box) != 4:
            return
        height, width = frame.shape[:2]
        x1, y1 = max(0, int(box[0] * width)), max(0, int(box[1] * height))
        x2, y2 = min(width, int(box[2] * width)), min(height, int(box[3] * height))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        quality = self._face_quality(crop, selected)
        landmarks = selected.get("landmarks", [])
        local_points = [[x * width - x1, y * height - y1] for x, y in landmarks]
        meta = {
            **selected,
            "quality": quality,
            "captured": time.time(),
            "local_landmarks": local_points,
        }
        with self.frame_lock:
            self.face_samples[selected["track_id"]] = (crop.copy(), meta)
            self.latest_face_crop, self.latest_face_meta = crop.copy(), meta

    @staticmethod
    def _face_quality(crop, selected):
        if crop.size == 0 or min(crop.shape[:2]) < 64:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        landmarks = selected.get("landmarks", [])
        # No recognition/enrollment from Haar boxes without alignment landmarks.
        if (
            len(landmarks) != 5
            or blur < 45
            or not 40 <= brightness <= 215
            or selected.get("confidence", 0) < 0.8
        ):
            return 0.0
        eyes = np.asarray(landmarks[:2], dtype=float)
        nose = np.asarray(landmarks[2], dtype=float)
        eye_distance = float(np.linalg.norm(eyes[0] - eyes[1]))
        if (
            eye_distance < 0.01
            or abs(float(eyes[0][1] - eyes[1][1])) > eye_distance * 0.35
        ):
            return 0.0
        if abs(float(nose[0] - np.mean(eyes[:, 0]))) > eye_distance * 0.4:
            return 0.0
        return round(min(1.0, blur / 180) * 0.35 + selected["confidence"] * 0.65, 3)

    def get_face_crop(self, min_quality=0.0, track_id=None):
        with self.frame_lock:
            samples = self.face_samples
            if track_id is None:
                if len(samples) != 1:
                    return None, None
                track_id = next(iter(samples))
            sample = samples.get(track_id)
            if sample is None:
                return None, None
            crop, meta = sample
            if time.time() - meta["captured"] > 1.0:
                return None, None
            if meta["quality"] < min_quality:
                return None, dict(meta)
            return crop.copy(), dict(meta)

    def snapshot_jpeg(
        self,
        quality=85,
    ):
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
            else:
                frame = None

        if frame is None:
            return None

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                int(quality),
            ],
        )

        if not ok:
            return None

        return encoded.tobytes()

    def _update_gaze(
        self,
        selected,
        face,
    ):
        if selected is not None:
            camera_x = (selected["center"][0] - 0.5) * 2.0

            camera_y = (selected["center"][1] - 0.5) * 2.0

            gx = -camera_x
            gy = camera_y

            self.gaze = (
                self.gaze[0] * 0.65 + gx * 0.35,
                self.gaze[1] * 0.65 + gy * 0.35,
            )

        else:
            self.gaze = (
                self.gaze[0] * 0.94,
                self.gaze[1] * 0.94,
            )

        face.set_gaze(*self.gaze)

    def _detect_objects(
        self,
        output,
    ):
        predictions = output[0].T

        found = {}

        for row in predictions:
            scores = row[4:]

            cls = int(np.argmax(scores))

            confidence = float(scores[cls])

            if confidence > 0.45 and cls < len(self.labels):
                label = self.labels[cls]

                previous = found.get(
                    label,
                    0.0,
                )

                if confidence > previous:
                    found[label] = confidence

        ordered = sorted(
            found.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [label for label, confidence in ordered[:8]]

    def stop(
        self,
    ):
        self.running = False
