#!/usr/bin/env python3
import base64
import json
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

ROS_AVAILABLE = True
ROS_ERROR = None
try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Int32
    from arm_msgs.msg import ArmJoint, ArmJoints
except Exception as exc:
    ROS_AVAILABLE = False
    ROS_ERROR = exc


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

        self.create_subscription(Image, os.getenv("MILO_ARM_COLOR_TOPIC", "/camera/color/image_raw"), self._color_cb, 2)
        self.create_subscription(Image, os.getenv("MILO_ARM_DEPTH_TOPIC", "/camera/depth/image_raw"), self._depth_cb, 2)

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

    def fresh(self, max_age=2.0):
        now = time.monotonic()
        with self.lock:
            return (
                self.color is not None and self.depth is not None and
                now - self.color_time <= max_age and now - self.depth_time <= max_age
            )


class RobotManager:
    """
    v3 scope:
      IDLE -> arm-mounted DaBai follows a face.
      "give me candy" -> tracking pauses, arm moves to a table-view pose,
      scans for a wrapped candy bar, reports detection/depth, and STOPS THERE.
      It intentionally does not grasp yet.

    This version does not auto-calibrate on boot.
    """

    def __init__(self):
        self.root = Path(os.getenv("JETSON_FRIEND_ROOT", Path(__file__).resolve().parents[2]))
        self.enabled = os.getenv("MILO_ARM_ENABLE", "1").lower() in {"1","true","yes","on"}
        self.face_follow = os.getenv("MILO_ARM_FACE_FOLLOW", "1").lower() in {"1","true","yes","on"}
        self.say_callback = None
        self.node = None
        self.executor = None

        self.stop_request = threading.Event()
        self.busy = threading.Event()
        self.armed = False
        self.mode = "SAFE"
        self.jobs = queue.Queue(maxsize=1)

        # Known logical limits from the custom STM32 firmware.
        self.limits = {1:(0.0,180.0),2:(0.0,180.0),3:(0.0,180.0),4:(0.0,180.0),5:(0.0,270.0),6:(30.0,180.0)}

        # We deliberately use only J1 (base yaw) + J4 (wrist pitch) for face following.
        # That is much safer than moving shoulder/elbow continuously just to centre a face.
        self.face_pan_joint = int(os.getenv("MILO_FACE_PAN_JOINT", "1"))
        self.face_tilt_joint = int(os.getenv("MILO_FACE_TILT_JOINT", "4"))
        self.face_pan_sign = float(os.getenv("MILO_FACE_PAN_SIGN", "-1"))
        self.face_tilt_sign = float(os.getenv("MILO_FACE_TILT_SIGN", "1"))
        self.face_step = float(os.getenv("MILO_FACE_STEP_DEG", "1.0"))
        self.face_deadband_x = float(os.getenv("MILO_FACE_DEADBAND_X", "0.10"))
        self.face_deadband_y = float(os.getenv("MILO_FACE_DEADBAND_Y", "0.12"))
        self.face_interval = float(os.getenv("MILO_FACE_INTERVAL_SEC", "0.30"))

        # Current *commanded* state. We do not pretend this is encoder feedback.
        # Face follow therefore begins only after a one-time HOME command below.
        self.home = self._pose("MILO_ARM_HOME_POSE", "90,90,90,90,135,120")
        self.table = self._pose("MILO_ARM_TABLE_POSE", "90,115,115,115,135,120")
        self.state = dict(self.home)

        # Search sweep is base-yaw only after moving to table pose.
        self.search_pan = [float(x) for x in os.getenv("MILO_CANDY_SCAN_J1", "65,78,90,102,115").split(",")]
        self.search_settle = float(os.getenv("MILO_CANDY_SCAN_SETTLE_SEC", "0.45"))

        self.face_detector = self._load_face_detector()
        self._face_lock = None
        self._face_candidate = None
        self._face_candidate_hits = 0
        self._face_last_seen = 0.0
        self._face_lock_required_hits = 5
        self._face_lock_max_jump = 0.18
        self._face_lock_lost_timeout = 2.5
        self._face_smooth_alpha = 0.30
        self._focus_tracker = None
        self._focus_tracker_kind = None
        self._focus_bbox_px = None
        self._focus_calibrated = False
        self._focus_calibrating = False
        self._focus_pan_sign = None
        self._focus_tilt_sign = None
        self._focus_center_j1 = None
        self._focus_center_j4 = None
        self._focus_j1_window = 35.0
        self._focus_j4_window = 12.0
        self._focus_err_x = 0.0
        self._focus_err_y = 0.0

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
        return {i+1: vals[i] for i in range(6)}

    def _load_face_detector(self):
        candidates = [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ]
        data = getattr(cv2, "data", None)
        if data is not None and getattr(data, "haarcascades", None):
            candidates.insert(0, data.haarcascades + "haarcascade_frontalface_default.xml")
        path = next((x for x in candidates if Path(x).exists()), None)
        if not path:
            print("[MILO ARM] Haar detector not found", flush=True)
            return None
        detector = cv2.CascadeClassifier(path)
        if detector.empty():
            print("[MILO ARM] Haar detector failed to load", flush=True)
            return None
        print(f"[MILO ARM] Arm-camera face detector: Haar {path}", flush=True)
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
        if not self.node:
            return
        msg = Int32()
        msg.data = 1 if on else 0
        self.node.torque_pub.publish(msg)
        time.sleep(0.1)

    def _startup_sequence(self):
        print("[MILO ARM] Waiting for DaBai RGB-D...", flush=True)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not self.stop_request.is_set():
            if self.node and self.node.fresh():
                break
            time.sleep(0.2)
        if not self.node or not self.node.fresh():
            print("[MILO ARM] RGB-D not ready; arm remains SAFE.", flush=True)
            return

        # Auto-ready, as requested.
        self.stop_request.clear()
        self.armed = True
        self.mode = "STARTING"
        self._torque(True)
        time.sleep(0.5)

        # Establish a known commanded pose once at boot. This is the reference for
        # all later incremental tracking. It is the previously used straight/home pose.
        print("[MILO ARM] Moving once to HOME reference pose.", flush=True)
        if not self._move_pose(self.home, 1800):
            self.mode = "STOPPED"
            return
        self.mode = "IDLE"
        print("[MILO ARM] READY: face tracking active.", flush=True)
        self._say("Arm ready. I am following your face.", "happy")

    def _can_move(self):
        return bool(self.enabled and self.armed and self.node and not self.stop_request.is_set())

    def _sleep(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.stop_request.is_set():
                return False
            time.sleep(min(0.05, end-time.monotonic()))
        return True

    def _move_joint(self, joint, angle, runtime_ms=300):
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

    def _box_center(self, box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _box_area(self, box):
        return max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))

    def _face_box(self, frame):
        faces = self._raw_faces(frame)
        now = time.monotonic()

        # Locked phase: keep the same face by proximity/scale continuity.
        if self._face_lock is not None:
            old = self._face_lock
            ocx, ocy = self._box_center(old)
            old_area = self._box_area(old)

            best = None
            best_score = 999.0
            for f in faces:
                cx, cy = self._box_center(f)
                dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
                ratio = self._box_area(f) / old_area
                if dist > 0.24:
                    continue
                if ratio < 0.35 or ratio > 2.8:
                    continue
                score = dist + 0.03 * abs(np.log(max(ratio, 1e-6)))
                if score < best_score:
                    best = f
                    best_score = score

            if best is None:
                if now - self._face_last_seen > 2.5:
                    print("[MILO ARM] FACE LOCK lost; waiting for your face again", flush=True)
                    self._face_lock = None
                    self._face_candidate = None
                    self._face_candidate_hits = 0
                    self._face_candidate_misses = 0
                return None

            a = 0.35
            smoothed = tuple((1.0 - a) * old[i] + a * best[i] for i in range(4))
            self._face_lock = smoothed
            self._face_last_seen = now
            return smoothed

        # Acquisition phase. Important: detections do NOT have to be on 5 consecutive
        # frames. Haar often misses alternate frames on this arm camera.
        if not hasattr(self, "_face_candidate_misses"):
            self._face_candidate_misses = 0

        if not faces:
            if self._face_candidate is not None:
                self._face_candidate_misses += 1
                if self._face_candidate_misses <= 8:
                    return None
                print("[MILO ARM] FACE ACQUIRE reset after too many missed frames", flush=True)
            self._face_candidate = None
            self._face_candidate_hits = 0
            self._face_candidate_misses = 0
            return None

        if self._face_candidate is None:
            cand = max(faces, key=self._box_area)
            self._face_candidate = cand
            self._face_candidate_hits = 1
            self._face_candidate_misses = 0
            print("[MILO ARM] FACE ACQUIRE 1/3 - keep looking at the arm camera", flush=True)
            return None

        pcx, pcy = self._box_center(self._face_candidate)
        prev_area = self._box_area(self._face_candidate)

        best = None
        best_dist = 999.0
        for f in faces:
            cx, cy = self._box_center(f)
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            ratio = self._box_area(f) / prev_area
            if dist <= 0.18 and 0.45 <= ratio <= 2.2 and dist < best_dist:
                best = f
                best_dist = dist

        if best is None:
            self._face_candidate_misses += 1
            if self._face_candidate_misses > 8:
                self._face_candidate = max(faces, key=self._box_area)
                self._face_candidate_hits = 1
                self._face_candidate_misses = 0
                print("[MILO ARM] FACE ACQUIRE restarted 1/3", flush=True)
            return None

        self._face_candidate = best
        self._face_candidate_hits += 1
        self._face_candidate_misses = 0
        print(f"[MILO ARM] FACE ACQUIRE {self._face_candidate_hits}/3", flush=True)

        if self._face_candidate_hits < 3:
            return None

        self._face_lock = self._face_candidate
        self._face_last_seen = now
        self._face_candidate = None
        self._face_candidate_hits = 0
        self._face_candidate_misses = 0
        cx, cy = self._box_center(self._face_lock)
        print(f"[MILO ARM] FACE LOCKED cx={cx:.2f} cy={cy:.2f}", flush=True)
        return self._face_lock

    def _make_cv_tracker(self):
        makers = [
            ("CSRT", lambda: cv2.legacy.TrackerCSRT_create() if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create") else None),
            ("KCF", lambda: cv2.legacy.TrackerKCF_create() if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create") else None),
            ("MOSSE", lambda: cv2.legacy.TrackerMOSSE_create() if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create") else None),
            ("CSRT", lambda: cv2.TrackerCSRT_create() if hasattr(cv2, "TrackerCSRT_create") else None),
            ("KCF", lambda: cv2.TrackerKCF_create() if hasattr(cv2, "TrackerKCF_create") else None),
        ]
        for name, fn in makers:
            try:
                tracker = fn()
                if tracker is not None:
                    return name, tracker
            except Exception:
                pass
        return None, None

    def _norm_to_px_bbox(self, box, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        x = int(max(0, min(w - 2, round(x1 * w))))
        y = int(max(0, min(h - 2, round(y1 * h))))
        bw = int(max(2, min(w - x, round((x2 - x1) * w))))
        bh = int(max(2, min(h - y, round((y2 - y1) * h))))
        return (x, y, bw, bh)

    def _px_to_norm_bbox(self, bbox, frame):
        h, w = frame.shape[:2]
        x, y, bw, bh = [float(v) for v in bbox]
        return (
            max(0.0, x / w),
            max(0.0, y / h),
            min(1.0, (x + bw) / w),
            min(1.0, (y + bh) / h),
        )

    def _start_focus_tracker(self, frame, box):
        name, tracker = self._make_cv_tracker()
        if tracker is None:
            print("[MILO ARM] No OpenCV object tracker available; continuing with detector lock", flush=True)
            self._focus_tracker = None
            self._focus_tracker_kind = None
            return False
        bbox = self._norm_to_px_bbox(box, frame)
        try:
            ok = tracker.init(frame, bbox)
            if ok is False:
                return False
            self._focus_tracker = tracker
            self._focus_tracker_kind = name
            self._focus_bbox_px = bbox
            print(f"[MILO ARM] FACE TARGET captured with {name} tracker", flush=True)
            return True
        except Exception as exc:
            print(f"[MILO ARM] Tracker init failed: {exc}", flush=True)
            self._focus_tracker = None
            return False

    def _tracked_face_box(self, frame):
        if self._focus_tracker is None:
            return None
        try:
            ok, bbox = self._focus_tracker.update(frame)
            if not ok:
                return None
            x, y, bw, bh = [float(v) for v in bbox]
            h, w = frame.shape[:2]
            if bw < 25 or bh < 25 or x + bw < 0 or y + bh < 0 or x >= w or y >= h:
                return None
            self._focus_bbox_px = bbox
            return self._px_to_norm_bbox(bbox, frame)
        except Exception:
            return None

    def _wait_face_center(self, samples=5, timeout=2.0):
        vals = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(vals) < samples and not self.stop_request.is_set():
            frame, _ = self.node.snapshot()
            box = self._tracked_face_box(frame) if frame is not None else None
            if box is None and frame is not None and self._face_lock is not None:
                box = self._face_lock
            if box is not None:
                vals.append(self._box_center(box))
            time.sleep(0.08)
        if not vals:
            return None
        return (
            float(np.median([v[0] for v in vals])),
            float(np.median([v[1] for v in vals])),
        )

    def _auto_calibrate_focus_axes(self):
        if self._focus_calibrated or self._focus_calibrating or not self._can_move():
            return self._focus_calibrated
        self._focus_calibrating = True
        try:
            print("[MILO ARM] FOCUS CALIBRATION: hold your head still for about 3 seconds", flush=True)
            base = self._wait_face_center(samples=5, timeout=2.0)
            if base is None:
                print("[MILO ARM] FOCUS CALIBRATION failed: face not stable", flush=True)
                return False

            j1 = float(self.state[1])
            j4 = float(self.state[4])
            self._focus_center_j1 = j1
            self._focus_center_j4 = j4

            if not self._move_joint(1, j1 + 2.0, 350):
                return False
            p1 = self._wait_face_center(samples=4, timeout=1.5)
            self._move_joint(1, j1, 350)
            self._sleep(0.2)

            if not self._move_joint(4, j4 + 2.0, 350):
                return False
            p4 = self._wait_face_center(samples=4, timeout=1.5)
            self._move_joint(4, j4, 350)
            self._sleep(0.2)

            if p1 is None or p4 is None:
                print("[MILO ARM] FOCUS CALIBRATION failed: tracker lost during probe", flush=True)
                return False

            dx = p1[0] - base[0]
            dy = p4[1] - base[1]
            if abs(dx) < 0.008 or abs(dy) < 0.008:
                print(f"[MILO ARM] FOCUS CALIBRATION failed: response too small dx={dx:+.3f} dy={dy:+.3f}", flush=True)
                return False

            self._focus_pan_sign = -1.0 if dx > 0 else 1.0
            self._focus_tilt_sign = -1.0 if dy > 0 else 1.0
            self._focus_calibrated = True
            print(
                f"[MILO ARM] FOCUS CALIBRATED dx/J1={dx:+.3f} dy/J4={dy:+.3f} "
                f"pan_sign={self._focus_pan_sign:+.0f} tilt_sign={self._focus_tilt_sign:+.0f}",
                flush=True,
            )
            return True
        finally:
            self._focus_calibrating = False
    def _face_loop(self):
        last_log = 0.0
        lost_since = None

        while True:
            try:
                if not (
                    self.enabled and self.armed and self.face_follow and
                    self.mode == "IDLE" and not self.busy.is_set() and
                    not self.stop_request.is_set() and self.node and self.node.fresh()
                ):
                    time.sleep(0.12)
                    continue

                frame, _ = self.node.snapshot()
                if frame is None:
                    time.sleep(0.1)
                    continue

                if self._focus_tracker is None and self._face_lock is None:
                    box = self._face_box(frame)
                    if self._face_lock is None:
                        time.sleep(0.08)
                        continue
                    box = self._face_lock
                    if self._start_focus_tracker(frame, box):
                        lost_since = None
                    time.sleep(0.08)
                    continue

                box = self._tracked_face_box(frame)
                if box is None:
                    if lost_since is None:
                        lost_since = time.monotonic()
                        print("[MILO ARM] FACE TARGET temporarily lost - arm frozen", flush=True)
                    if time.monotonic() - lost_since > 2.0:
                        print("[MILO ARM] FACE TARGET lost - reacquiring", flush=True)
                        self._focus_tracker = None
                        self._focus_tracker_kind = None
                        self._face_lock = None
                        self._face_candidate = None
                        self._face_candidate_hits = 0
                        self._focus_calibrated = False
                        lost_since = None
                    time.sleep(0.1)
                    continue

                lost_since = None

                if not self._focus_calibrated:
                    if not self._auto_calibrate_focus_axes():
                        time.sleep(0.3)
                        continue
                    time.sleep(0.1)
                    continue

                cx, cy = self._box_center(box)
                ex = cx - 0.5
                ey = cy - 0.5

                a = 0.35
                self._focus_err_x = (1.0 - a) * self._focus_err_x + a * ex
                self._focus_err_y = (1.0 - a) * self._focus_err_y + a * ey
                fx = self._focus_err_x
                fy = self._focus_err_y

                pan_delta = 0.0
                tilt_delta = 0.0
                if abs(fx) > 0.06:
                    pan_delta = self._focus_pan_sign * np.sign(fx) * min(1.8, max(0.6, abs(fx) * 7.0))
                if abs(fy) > 0.08:
                    tilt_delta = self._focus_tilt_sign * np.sign(fy) * min(1.0, max(0.5, abs(fy) * 4.0))

                j1_min = max(self.limits[1][0], self._focus_center_j1 - self._focus_j1_window)
                j1_max = min(self.limits[1][1], self._focus_center_j1 + self._focus_j1_window)
                j4_min = max(self.limits[4][0], self._focus_center_j4 - self._focus_j4_window)
                j4_max = min(self.limits[4][1], self._focus_center_j4 + self._focus_j4_window)

                if time.monotonic() - last_log > 0.6:
                    print(
                        f"[MILO ARM] FOCUS cx={cx:.2f} cy={cy:.2f} "
                        f"err=({fx:+.2f},{fy:+.2f}) "
                        f"dJ1={pan_delta:+.1f} dJ4={tilt_delta:+.1f}",
                        flush=True,
                    )
                    last_log = time.monotonic()

                if pan_delta:
                    target1 = max(j1_min, min(j1_max, self.state[1] + pan_delta))
                    self._move_joint(1, target1, 260)

                if tilt_delta and not self.stop_request.is_set():
                    target4 = max(j4_min, min(j4_max, self.state[4] + tilt_delta))
                    self._move_joint(4, target4, 260)

                time.sleep(0.08)

            except Exception as exc:
                print(f"[MILO ARM] face focus recovery: {exc}", flush=True)
                time.sleep(0.5)
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

    def handle_voice(self,text):
        cmd=parse_robot_command(text)
        if not cmd:
            return RobotResult(False)
        action=cmd["action"]

        if action=="stop":
            self.stop_request.set()
            self.armed=False
            self.face_follow=False
            self.mode="STOPPED"
            try:self._torque(False)
            except Exception:pass
            return RobotResult(True,"Arm stopped and torque disabled.","concerned")

        if action=="arm_ready":
            if not self.node:
                return RobotResult(True,"The arm controller is not available.","concerned")
            self.stop_request.clear()
            self.armed=True
            self._torque(True)
            self.mode="IDLE"
            return RobotResult(True,"Arm ready.","neutral")

        if action=="tracking_on":
            if not self.armed:
                return RobotResult(True,"The arm is stopped.","concerned")
            self.face_follow=True
            self.mode="IDLE"
            return RobotResult(True,"I am following your face.","happy")

        if action=="tracking_off":
            self.face_follow=False
            return RobotResult(True,"Face tracking is off.","neutral")

        if action=="find_candy":
            if not self.armed or self.stop_request.is_set():
                return RobotResult(True,"The arm is stopped.","concerned")
            if self.busy.is_set() or not self.jobs.empty():
                return RobotResult(True,"I am already moving the arm.","neutral")
            self.face_follow=False
            self.jobs.put_nowait("find_candy")
            return RobotResult(True,"I will look for the candy on the table.","thinking")

        return RobotResult(False)

    def close(self):
        self.stop_request.set()
        try:self.jobs.put_nowait(None)
        except Exception:pass
        try:
            if self.executor:self.executor.shutdown(timeout_sec=1.0)
        except Exception:pass
        try:
            if self.node:self.node.destroy_node()
        except Exception:pass
