#!/usr/bin/env python3
import base64
import fcntl
import json
import math
import os
import queue
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .face_search import FaceSearch, orient_face_image

ROS_AVAILABLE = True
ROS_ERROR = None
try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Int32
    from arm_msgs.msg import ArmJoint, ArmJoints
except Exception as exc:
    ROS_AVAILABLE = False
    ROS_ERROR = exc
    Node = object  # Allow the rest of MILO to run without ROS installed.


@dataclass
class RobotResult:
    handled: bool
    text: str = ""
    mood: str = "neutral"


def _normalize(text):
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\s'-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_robot_command(text):
    v = _normalize(text)

    # Emergency stop gets priority over every fuzzy phrase.
    if re.search(r"\b(stop arm|emergency stop|freeze arm|stop moving|arm stop)\b", v):
        return {"action": "stop"}

    if re.search(r"\b(face tracking off|stop face tracking|don't follow me|do not follow me)\b", v):
        return {"action": "tracking_off"}
    if re.search(r"\b(face tracking on|follow my face|follow me|track my face)\b", v):
        return {"action": "tracking_on"}

    # Whisper base.en commonly confuses "candy" with candid/candie.
    candy_word = r"(?:candy|candie|candid|candies)"
    if re.search(rf"\b(?:give|bring|get|find|look for|search for)\b.*\b{candy_word}\b", v):
        return {"action": "find_candy", "target": "wrapped candy bar"}

    if re.search(r"\b(arm(?: is)? ready|enable arm|resume arm)\b", v):
        return {"action": "arm_ready"}

    return None


class ArmCameraNode(Node):
    def __init__(self):
        super().__init__("milo_arm_camera")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.color = None
        self.depth = None
        self.color_time = 0.0
        self.depth_time = 0.0

        self.create_subscription(Image, os.getenv("MILO_ARM_COLOR_TOPIC", "/camera/color/image_raw"), self._color_cb, qos_profile_sensor_data)
        self.create_subscription(Image, os.getenv("MILO_ARM_DEPTH_TOPIC", "/camera/depth/image_raw"), self._depth_cb, qos_profile_sensor_data)

        self.joint_pub = self.create_publisher(ArmJoint, os.getenv("MILO_ARM_JOINT_TOPIC", "/arm_joint"), 10)
        self.joints_pub = self.create_publisher(ArmJoints, os.getenv("MILO_ARM_JOINTS_TOPIC", "/arm6_joints"), 10)
        self.torque_pub = self.create_publisher(Int32, os.getenv("MILO_ARM_TORQUE_TOPIC", "/arm_torque"), 10)

    def _color_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.color = frame
                self.color_time = time.monotonic()
        except Exception as exc:
            print(f"[MILO ARM] color conversion error: {exc}", flush=True)

    def _depth_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.depth = np.asarray(frame)
                self.depth_time = time.monotonic()
        except Exception as exc:
            print(f"[MILO ARM] depth conversion error: {exc}", flush=True)

    def snapshot(self):
        with self.lock:
            color = None if self.color is None else self.color.copy()
            depth = None if self.depth is None else self.depth.copy()
            return color, depth

    def face_snapshot(self):
        with self.lock:
            return (None if self.color is None else self.color.copy(), self.color_time)

    def fresh(self, max_age=0.75):
        now = time.monotonic()
        with self.lock:
            return (
                self.color is not None and self.depth is not None and
                now - self.color_time <= max_age and now - self.depth_time <= max_age
            )


class RobotManager:
    """Bounded face search; joint state records commands, never measurements."""

    def __init__(self):
        self.root = Path(os.getenv("JETSON_FRIEND_ROOT", Path(__file__).resolve().parents[2]))
        self.enabled = os.getenv("MILO_ARM_ENABLE", "1").lower() in {"1","true","yes","on"}
        self.face_follow = os.getenv("MILO_ARM_FACE_FOLLOW", "1").lower() in {"1","true","yes","on"}
        self.say_callback = None
        self.node = None
        self.executor = None

        self.shutdown_request = threading.Event()
        self.stop_request = threading.Event()
        self.busy = threading.Event()
        self.armed = False
        self.mode = "SAFE"
        self.jobs = queue.Queue(maxsize=1)

        # Known logical limits from the custom STM32 firmware.
        self.limits = {1:(0.0,180.0),2:(0.0,180.0),3:(0.0,180.0),4:(0.0,180.0),5:(0.0,270.0),6:(30.0,180.0)}

        self.motion_lock = threading.RLock()
        self.startup_block = os.getenv("MILO_ARM_STARTUP_BLOCK_REASON", "")
        self.face_image_rotation = int(os.getenv("MILO_FACE_IMAGE_ROTATION_DEG", "0"))
        orient_face_image(None, self.face_image_rotation)  # Validate even before frames arrive.
        self.face_search = FaceSearch(
            minimum=float(os.getenv("MILO_FACE_SEARCH_J1_MIN", "45")),
            maximum=float(os.getenv("MILO_FACE_SEARCH_J1_MAX", "135")),
            step=float(os.getenv("MILO_FACE_SEARCH_STEP_DEG", "3")),
            runtime_ms=int(os.getenv("MILO_FACE_SEARCH_RUNTIME_MS", "320")),
            settle=float(os.getenv("MILO_FACE_SEARCH_SETTLE_SEC", "0.10")),
            direction=int(os.getenv("MILO_FACE_SEARCH_DIRECTION", "1")),
            required_hits=int(os.getenv("MILO_FACE_SEARCH_CONFIRMATIONS", "3")),
            loss_timeout=float(os.getenv("MILO_FACE_SEARCH_LOSS_TIMEOUT_SEC", "2.5")),
        )
        # J1-only face following. J2..J6 stay at the operator-calibrated face pose.
        self.face_follow_deadband_x = float(os.getenv("MILO_FACE_FOLLOW_DEADBAND_X", "0.06"))
        self.face_follow_target_x = float(os.getenv("MILO_FACE_FOLLOW_TARGET_X", "0.36"))
        self.face_follow_step = float(os.getenv("MILO_FACE_FOLLOW_STEP_DEG", "1"))
        self.face_follow_runtime_ms = int(os.getenv("MILO_FACE_FOLLOW_RUNTIME_MS", "320"))
        self.face_follow_settle = float(os.getenv("MILO_FACE_FOLLOW_SETTLE_SEC", "0.12"))
        self.face_follow_sign = int(os.getenv("MILO_FACE_FOLLOW_J1_SIGN", "1"))
        self.face_follow_reverse_margin = float(os.getenv("MILO_FACE_FOLLOW_REVERSE_MARGIN", "0.025"))
        if not 0.03 <= self.face_follow_deadband_x <= 0.20:
            raise ValueError("Face-follow deadband must be 0.03..0.20")
        if not 0.20 <= self.face_follow_target_x <= 0.80:
            raise ValueError("MILO_FACE_FOLLOW_TARGET_X must be 0.20..0.80")
        if self.face_follow_step != 1:
            raise ValueError("Initial face-follow step is intentionally fixed at 1 degree")
        if not 320 <= self.face_follow_runtime_ms <= 1200:
            raise ValueError("Face-follow runtime must be 320..1200 ms")
        if not 0.10 <= self.face_follow_settle <= 1.0:
            raise ValueError("Face-follow settle must be 0.10..1.0 s")
        if self.face_follow_sign not in (-1, 1):
            raise ValueError("MILO_FACE_FOLLOW_J1_SIGN must be -1 or +1")
        if not 0.01 <= self.face_follow_reverse_margin <= 0.08:
            raise ValueError("Face-follow reverse margin must be 0.01..0.08")
        self.face_follow_sign_verified = False
        self.face_follow_previous_error = None
        self.face_follow_previous_delta = None

        self.face_pose = self._pose("MILO_ARM_FACE_SEARCH_POSE", "90,115,115,110,135,120")
        if not self.face_search.minimum <= self.face_pose[1] <= self.face_search.maximum:
            raise ValueError("Face-search pose J1 must be inside the search sector")
        self.test_duration = float(os.getenv("MILO_FACE_SEARCH_TEST_DURATION_SEC", "0"))
        if not math.isfinite(self.test_duration) or not 0 <= self.test_duration <= 120:
            raise ValueError("Supervised test duration must be 0 (disabled)..120 seconds")
        self.test_deadline = None
        self.pose_ready = False
        self.next_search_move = 0.0
        # Current *commanded* state. We do not pretend this is encoder feedback.
        # Search begins only after the configured test pose has been commanded.
        self.home = self._pose("MILO_ARM_HOME_POSE", "90,90,90,90,135,120")
        self.table = self._pose("MILO_ARM_TABLE_POSE", "90,115,115,115,135,120")
        self.state = dict(self.home)

        # Search sweep is base-yaw only after moving to table pose.
        self.search_pan = [float(x) for x in os.getenv("MILO_CANDY_SCAN_J1", "65,78,90,102,115").split(",")]
        self.search_settle = float(os.getenv("MILO_CANDY_SCAN_SETTLE_SEC", "0.45"))

        self.face_pose_runtime = int(os.getenv("MILO_FACE_SEARCH_POSE_RUNTIME_MS", "5000"))
        if not 2500 <= self.face_pose_runtime <= 5000:
            raise ValueError("Face-search pose runtime must be 2500..5000 ms")
        self.controller_lock = None
        if self.enabled:
            self.controller_lock = open(f"/tmp/milo-arm-controller-{os.getenv('ROS_DOMAIN_ID', '30')}.lock", "a")
            try:
                fcntl.flock(self.controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.enabled = False
                print("[MILO ARM] SAFE: another MILO controller owns the arm", flush=True)
        self.face_detector = self._load_face_detector()
        if self.enabled:
            self._start_ros()
            threading.Thread(target=self._worker_loop, name="milo-arm-worker", daemon=True).start()
            threading.Thread(target=self._face_loop, name="milo-arm-face", daemon=True).start()
            threading.Thread(target=self._startup_sequence, name="milo-arm-startup", daemon=True).start()

    def bind_speech(self, callback):
        self.say_callback = callback

    def _say(self, text, mood="neutral"):
        if self.say_callback:
            try:
                self.say_callback(text, mood)
            except Exception:
                pass

    def _pose(self, key, default):
        vals = [float(x.strip()) for x in os.getenv(key, default).split(",")]
        if len(vals) != 6:
            raise ValueError(f"{key} must contain 6 comma-separated joint angles")
        for j, value in enumerate(vals, 1):
            lo, hi = self.limits[j]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{key}: J{j} must be finite and within {lo}..{hi}")
        return {i+1: vals[i] for i in range(6)}

    def _load_face_detector(self):
        # Prefer YuNet. The arm camera often sees a three-quarter/profile face;
        # the old frontal Haar cascade is unreliable for that view.
        model = Path(os.getenv(
            "MILO_FACE_MODEL",
            os.getenv("FACE_MODEL", str(self.root / "models" / "vision" / "face_detection_yunet.onnx"))
        ))
        score = float(os.getenv("MILO_FACE_YUNET_SCORE", "0.55"))
        if not 0.3 <= score <= 0.95:
            raise ValueError("MILO_FACE_YUNET_SCORE must be 0.3..0.95")

        if hasattr(cv2, "FaceDetectorYN") and model.exists():
            try:
                detector = cv2.FaceDetectorYN.create(
                    str(model), "", (320, 320), score, 0.3, 5000
                )
                print(
                    f"[MILO ARM] Arm-camera face detector: YuNet {model} "
                    f"(score>={score:.2f})",
                    flush=True,
                )
                return ("yunet", detector)
            except Exception as exc:
                print(f"[MILO ARM] YuNet load failed: {exc}; trying Haar fallback", flush=True)
        else:
            reason = "OpenCV FaceDetectorYN unavailable" if not hasattr(cv2, "FaceDetectorYN") else f"model missing: {model}"
            print(f"[MILO ARM] YuNet unavailable ({reason}); trying Haar fallback", flush=True)

        candidates = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ]
        data = getattr(cv2, "data", None)
        if data is not None and getattr(data, "haarcascades", None):
            candidates.insert(0, data.haarcascades + "haarcascade_frontalface_default.xml")
        path = next((x for x in candidates if Path(x).exists()), None)
        if not path:
            print("[MILO ARM] No usable face detector found", flush=True)
            return None
        detector = cv2.CascadeClassifier(path)
        if detector.empty():
            print("[MILO ARM] Haar detector failed to load", flush=True)
            return None
        print(f"[MILO ARM] Arm-camera face detector FALLBACK: Haar {path}", flush=True)
        return ("haar", detector)

    def _start_ros(self):
        if not ROS_AVAILABLE:
            print(f"[MILO ARM] ROS unavailable: {ROS_ERROR}", flush=True)
            self.enabled = False
            return
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = ArmCameraNode()
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, name="milo-arm-ros", daemon=True).start()

    def _torque(self, on):
        if not self.node or (on and not self._can_move()):
            return
        msg = Int32()
        msg.data = 1 if on else 0
        self.node.torque_pub.publish(msg)
        time.sleep(0.1)

    def _arm_available(self):
        return bool(self.node and all(pub.get_subscription_count() > 0 for pub in
                    (self.node.joint_pub, self.node.joints_pub, self.node.torque_pub)))

    def _startup_sequence(self):
        if self.startup_block:
            print(f"[MILO ARM] SAFE: {self.startup_block}", flush=True)
            return
        print("[MILO ARM] Waiting for arm endpoints and fresh DaBai RGB-D...", flush=True)
        deadline = time.monotonic() + float(os.getenv("MILO_ARM_READY_TIMEOUT_SEC", "45"))
        while not self.stop_request.is_set():
            if self._arm_available() and self.node.fresh() and self.face_detector:
                break
            if time.monotonic() >= deadline:
                print("[MILO ARM] SAFE: arm endpoints, fresh RGB-D, or face detector unavailable", flush=True)
                return
            self.stop_request.wait(0.1)
        with self.motion_lock:
            if self.stop_request.is_set() or not self.enabled:
                return
            self.armed = True
            self.mode = "FACE_SEARCH_POSE" if self.face_follow else "IDLE"

    def _can_move(self):
        return bool(self.enabled and self.armed and not self.startup_block and
                    self._arm_available() and self.node.fresh() and
                    not self.stop_request.is_set())

    def _sleep(self, seconds):
        end = time.monotonic() + max(0.0, seconds)
        while True:
            if self.stop_request.is_set():
                return False
            remaining = end - time.monotonic()
            if remaining <= 0:
                return True
            self.stop_request.wait(min(0.05, remaining))

    def _move_joint(self, joint, angle, runtime_ms=300):
        with self.motion_lock:
            if not self._can_move():
                return False
            low, high = self.limits[joint]
            angle = max(low, min(high, float(angle)))
            msg = ArmJoint()
            msg.id = joint
            msg.joint = int(round(angle))
            msg.time = max(100, min(5000, int(runtime_ms)))
            self.node.joint_pub.publish(msg)
            self.state[joint] = angle
            return self._sleep(msg.time/1000.0 + 0.06)

    def _move_pose(self, pose, runtime_ms=1300):
        with self.motion_lock:
            if not self._can_move():
                return False
            vals=[]
            for j in range(1,7):
                lo,hi=self.limits[j]
                vals.append(max(lo,min(hi,float(pose[j]))))
            msg=ArmJoints()
            msg.joint1,msg.joint2,msg.joint3,msg.joint4,msg.joint5,msg.joint6=[int(round(v)) for v in vals]
            msg.time=max(100,min(5000,int(runtime_ms)))
            self.node.joints_pub.publish(msg)
            self.state={i+1:vals[i] for i in range(6)}
            return self._sleep(msg.time/1000.0 + 0.08)

    def _raw_faces(self, frame):
        if frame is None or self.face_detector is None:
            return []
        frame = orient_face_image(frame, self.face_image_rotation)
        h, w = frame.shape[:2]
        kind, detector = self.face_detector

        if kind == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.06,
                minNeighbors=4,
                minSize=(35, 35),
            )
            out = []
            for x, y, bw, bh in faces:
                if bw < 35 or bh < 35:
                    continue
                if bw > 0.80 * w or bh > 0.80 * h:
                    continue
                x1, y1 = x / w, y / h
                x2, y2 = (x + bw) / w, (y + bh) / h
                out.append((x1, y1, x2, y2))
            return out

        try:
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            if faces is None:
                return []
            out = []
            for f in faces:
                x, y, bw, bh = [float(v) for v in f[:4]]
                out.append((max(0.0, x / w), max(0.0, y / h), min(1.0, (x + bw) / w), min(1.0, (y + bh) / h)))
            return out
        except Exception:
            return []

    def _face_tick(self):
        with self.motion_lock:
            if self.test_deadline is not None and time.monotonic() >= self.test_deadline:
                self.stop_request.set()
                self.armed = False
                self.mode = "STOPPED"
                self.startup_block = "supervised test completed; restart required for another test"
                self.test_deadline = None
                print("[MILO ARM] STOPPED: supervised test time limit; no further commands", flush=True)
                return
            if not (self.face_follow and not self.busy.is_set() and
                    self.mode in {"FACE_SEARCH_POSE", "FACE_SEARCH", "FACE_FOUND", "FACE_HOLD"}):
                return
            if not self._can_move():
                # Connectivity loss latches SAFE. A returning stream never resumes motion itself.
                if self.armed:
                    self.armed = False
                    self.mode = "SAFE"
                    self.pose_ready = False
                    self.face_search.reset()
                    self.face_follow_sign_verified = False
                    self.face_follow_previous_error = None
                    self.face_follow_previous_delta = None
                    print("[MILO ARM] SAFE: arm endpoint or fresh RGB-D lost; resume required", flush=True)
                return
            if not self.face_detector:
                self.armed = False
                self.mode = "SAFE"
                return
            if not self.pose_ready:
                self.mode = "FACE_SEARCH_POSE"
                print(f"[MILO ARM] FACE_SEARCH_POSE test command: {self.face_pose}", flush=True)
                if self.test_duration:
                    self.test_deadline = time.monotonic() + self.test_duration
                self._torque(True)
                if not self._move_pose(self.face_pose, self.face_pose_runtime):
                    self.armed = False
                    self.mode = "STOPPED"
                    return
                self.pose_ready = True
                self.face_search.reset()
                self.face_follow_sign_verified = False
                self.face_follow_previous_error = None
                self.face_follow_previous_delta = None
                self.next_search_move = time.monotonic() + self.face_search.settle
                self.mode = "FACE_SEARCH"
                print("[MILO ARM] FACE_SEARCH: bounded J1 sweep enabled", flush=True)
                return
            frame, stamp = self.node.face_snapshot()
            now = time.monotonic()
            if frame is None or stamp <= (self.face_search.last_frame or 0):
                return
            faces = self._raw_faces(frame)
            # Detection can take time: recheck freshness before any publish.
            if not self._can_move() or time.monotonic() - stamp > 0.75:
                return
            previous = self.mode
            observed_at = time.monotonic()
            self.mode = self.face_search.observe(faces, stamp, observed_at)
            if self.mode != previous:
                print(f"[MILO ARM] {self.mode}: " +
                      ("face confirmed; J1 follow enabled" if self.mode == "FACE_FOUND"
                       else "face lost after timeout; bounded reacquisition enabled"), flush=True)

            if self.mode == "FACE_FOUND":
                # FACE_FOUND can persist briefly after a missed detector frame.
                # Never move from a stale box: follow only a face matched THIS frame.
                if self.face_search.last_seen != observed_at or self.face_search.box is None:
                    return
                if now < self.next_search_move:
                    return

                x1, _, x2, _ = self.face_search.box
                cx = (x1 + x2) / 2.0

                # Calibrated from the physically verified DaBai frame.
                # The desired face position is x ~= 0.36, not the geometric image center.
                error = cx - self.face_follow_target_x

                # Fixed from observed real motion:
                # decreasing J1 moved the face right in image;
                # increasing J1 moved the face left.
                # So sign=+1 is correct. Never auto-flip from one noisy detection.
                self.face_follow_sign = 1
                self.face_follow_sign_verified = True
                self.face_follow_previous_error = None
                self.face_follow_previous_delta = None

                if abs(error) <= self.face_follow_deadband_x:
                    if self.mode != "FACE_HOLD":
                        print(
                            f"[MILO ARM] FACE_HOLD cx={cx:.3f} target={self.face_follow_target_x:.3f} "
                            f"J1={int(round(self.state[1]))}",
                            flush=True,
                        )
                    self.mode = "FACE_HOLD"
                    return

                self.mode = "FACE_FOUND"

                direction = 1 if error > 0 else -1
                delta = self.face_follow_sign * direction * self.face_follow_step
                target = max(self.face_search.minimum,
                             min(self.face_search.maximum, self.state[1] + delta))
                if target == self.state[1]:
                    return

                msg = ArmJoint()
                msg.id = 1
                msg.joint = int(round(target))
                msg.time = self.face_follow_runtime_ms
                if self.stop_request.is_set():
                    return
                self.node.joint_pub.publish(msg)
                self.face_follow_previous_error = error
                self.face_follow_previous_delta = delta
                self.state[1] = msg.joint  # Commanded, NOT measured.
                self.next_search_move = (
                    time.monotonic() + msg.time / 1000 + self.face_follow_settle
                )
                print(
                    f"[MILO ARM] FACE_FOLLOW cx={cx:.3f} err={error:+.3f} "
                    f"J1={msg.joint} sign={self.face_follow_sign:+d}",
                    flush=True,
                )
                return

            if self.face_search.hits:
                # During acquisition, freeze search until face confirmation.
                return

            self.face_follow_previous_error = None
            self.face_follow_previous_delta = None
            if now < self.next_search_move:
                return
            target = self.face_search.next_target(self.state[1])
            msg = ArmJoint()
            msg.id, msg.joint, msg.time = 1, int(round(target)), self.face_search.runtime_ms
            if self.stop_request.is_set():
                return
            self.node.joint_pub.publish(msg)
            self.state[1] = msg.joint  # Commanded, NOT measured.
            self.next_search_move = time.monotonic() + msg.time / 1000 + self.face_search.settle
            print(f"[MILO ARM] FACE_SEARCH J1 command={msg.joint} runtime_ms={msg.time}", flush=True)

    def _face_loop(self):
        while not self.shutdown_request.wait(0.08):
            try:
                self._face_tick()
            except Exception as exc:
                with self.motion_lock:
                    self.armed = False
                    self.mode = "SAFE"
                    self.pose_ready = False
                print(f"[MILO ARM] SAFE: face search error: {exc}", flush=True)
        # Emergency stop does not terminate the service; explicit resume may restart search.

    def _gemma_box(self, frame, target):
        ok,jpg=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),82])
        if not ok:
            return None
        image64=base64.b64encode(jpg.tobytes()).decode("ascii")
        prompt=(
            f'Locate the most obvious {target} in the image. '
            'Return ONLY JSON with key box_2d in [ymin,xmin,ymax,xmax] order, '
            'each coordinate as an integer from 0 to 1000. '
            'If it is not visible return {"box_2d":null}.'
        )
        payload={
            "messages":[{"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+image64}}
            ]}],
            "temperature":0.0,
            "max_tokens":128,
            "stream":False,
        }
        url=os.getenv("LLAMA_SERVER_URL","http://127.0.0.1:8081/v1/chat/completions")
        req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(req,timeout=float(os.getenv("VLM_TIMEOUT_SEC","90"))) as r:
                data=json.loads(r.read().decode())
            msg=data.get("choices",[{}])[0].get("message",{})
            raw=msg.get("content","")
            if isinstance(raw,list):
                raw="".join(str(x.get("text","")) for x in raw if isinstance(x,dict))
            raw=str(raw or msg.get("reasoning_content","") or "").strip()
            if not raw:
                print("[MILO ARM] Gemma empty response:",json.dumps(data,ensure_ascii=False)[:1500],flush=True)
                return None
            m=re.search(r"\{.*\}",raw,re.S)
            obj=json.loads(m.group(0) if m else raw)
            box=obj.get("box_2d")
            if not box or len(box)!=4:
                return None
            y1,x1,y2,x2=[max(0,min(1000,float(v))) for v in box]
            if x2<=x1 or y2<=y1:
                return None
            return (x1/1000.0,y1/1000.0,x2/1000.0,y2/1000.0)
        except Exception as exc:
            print(f"[MILO ARM] Gemma detection failed: {exc}",flush=True)
            return None

    def _depth_at_box(self, box):
        color,depth=self.node.snapshot()
        if color is None or depth is None or box is None:
            return None
        dh,dw=depth.shape[:2]
        x1,y1,x2,y2=box
        cx=(x1+x2)/2.0
        cy=(y1+y2)/2.0
        px=int(np.clip(round(cx*(dw-1)),0,dw-1))
        py=int(np.clip(round(cy*(dh-1)),0,dh-1))
        rx=max(4,int((x2-x1)*dw/8))
        ry=max(4,int((y2-y1)*dh/8))
        patch=np.asarray(depth[max(0,py-ry):min(dh,py+ry+1),max(0,px-rx):min(dw,px+rx+1)],dtype=np.float32)
        vals=patch[np.isfinite(patch)&(patch>0)]
        if vals.size<8:
            return None
        z=float(np.median(vals))
        if z>20:
            z/=1000.0
        return z if 0.12<=z<=2.0 else None

    def _find_candy(self):
        self.busy.set()
        self.face_follow=False
        self.mode="TABLE_MOVE"
        try:
            self._say("I am looking for the candy.", "thinking")
            print("[MILO ARM] Moving to TABLE VIEW pose.",flush=True)
            if not self._move_pose(self.table,1600):
                return RobotResult(True,"Candy search stopped.","concerned")
            self.mode="CANDY_SEARCH"

            for pan in self.search_pan:
                if self.stop_request.is_set():
                    return RobotResult(True,"Candy search stopped.","concerned")
                if not self._move_joint(1,pan,450):
                    return RobotResult(True,"Candy search stopped.","concerned")
                if not self._sleep(self.search_settle):
                    return RobotResult(True,"Candy search stopped.","concerned")
                frame,_=self.node.snapshot()
                if frame is None:
                    continue
                box=self._gemma_box(frame,"wrapped candy bar such as a Snickers bar")
                if box:
                    z=self._depth_at_box(box)
                    self.mode="CANDY_FOUND"
                    if z is not None:
                        print(f"[MILO ARM] Candy found at ~{z:.2f} m.",flush=True)
                        return RobotResult(True,f"I found the candy about {z:.2f} meters from my arm camera.","happy")
                    print("[MILO ARM] Candy found; depth unavailable.",flush=True)
                    return RobotResult(True,"I found the candy.","happy")

            self.mode="CANDY_NOT_FOUND"
            return RobotResult(True,"I could not see the candy on the table.","concerned")
        finally:
            self.busy.clear()
            # Stay in table view after a search. Do not suddenly swing back toward the face.
            if not self.stop_request.is_set() and self.mode not in {"CANDY_FOUND","CANDY_NOT_FOUND"}:
                self.mode="IDLE"

    def _worker_loop(self):
        while True:
            job=self.jobs.get()
            if job is None:
                return
            try:
                if job=="find_candy":
                    result=self._find_candy()
                    if result.text:
                        self._say(result.text,result.mood)
            except Exception as exc:
                print(f"[MILO ARM] action error: {exc}",flush=True)
                self._say("The arm action stopped safely.","concerned")
            finally:
                self.jobs.task_done()

    def handle_voice(self, text):
        # Signal first so an in-flight synchronous wait can exit before acquiring the lock.
        if (parse_robot_command(text) or {}).get("action") == "stop":
            self.stop_request.set()
        with self.motion_lock:
            return self._handle_voice(text)

    def _handle_voice(self,text):
        cmd=parse_robot_command(text)
        if not cmd:
            return RobotResult(False)
        action=cmd["action"]

        if action=="stop":
            self.stop_request.set()
            self.armed=False
            self.face_follow=False
            self.mode="STOPPED"
            self.pose_ready=False
            try:self._torque(False)
            except Exception:pass
            return RobotResult(True,"Arm stopped and torque disabled.","concerned")

        if action in {"arm_ready", "tracking_on"}:
            if self.startup_block or not self._arm_available() or not self.node.fresh() or not self.face_detector:
                return RobotResult(True, "Arm stays safe: startup blocked, controller, camera, or detector unavailable.", "concerned")
            if self.busy.is_set() or not self.jobs.empty():
                return RobotResult(True, "An arm action is already active.", "neutral")
            self.stop_request.clear()
            self.armed = True
            self.face_follow = True
            self.face_search.reset()
            self.face_follow_sign_verified = False
            self.face_follow_previous_error = None
            self.face_follow_previous_delta = None
            self.mode = "FACE_SEARCH" if self.pose_ready else "FACE_SEARCH_POSE"
            return RobotResult(True, "Face tracking enabled.", "neutral")

        if action=="tracking_off":
            self.face_follow=False
            return RobotResult(True,"Face tracking is off.","neutral")

        if action=="find_candy":
            return RobotResult(True, "Candy search is disabled while we validate face search.", "neutral")

        return RobotResult(False)

    def close(self):
        self.shutdown_request.set()
        self.stop_request.set()
        self.armed = False
        try:self.jobs.put_nowait(None)
        except Exception:pass
        try:
            if self.executor:self.executor.shutdown(timeout_sec=1.0)
        except Exception:pass
        try:
            if self.node:self.node.destroy_node()
        except Exception:pass
        if self.controller_lock:
            self.controller_lock.close()
