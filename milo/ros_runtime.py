from __future__ import annotations

import threading
import time

from .safety import ArmSafetyGate


class RosRuntime:
    def __init__(self, cfg):
        self.cfg = cfg
        self.safety = ArmSafetyGate(cfg)
        self.latest_color = None
        self.color_stamp = 0.0
        self.latest_depth = None
        self.depth_stamp = 0.0
        self.node = None
        self.executor = None
        self.thread = None
        self.available = False
        self._rclpy = None
        self._owns_context = False
        self._subs = []
        self._decode_warnings = set()
        self._last_camera_source = "none"
        self._command_subscription_ready_logged = False
        self._pending_motion = {}
        self.latest_raw = {}
        self.latest_feedback = {}
        self.feedback_stamp = 0.0
        self._last_raw_warning = {}

    def start(self):
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image, CompressedImage
            from arm_msgs.msg import ArmJoint, ArmJoints
            try:
                from cv_bridge import CvBridge
            except Exception:
                CvBridge = None
        except Exception as exc:
            print(f"[ROS] unavailable: {exc}", flush=True)
            return

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_context = True
        outer = self

        camera_qos = qos_profile_sensor_data

        class NodeImpl(Node):
            def __init__(self):
                super().__init__("milo_clean_rc12")
                self.bridge = CvBridge() if CvBridge is not None else None
                self.pub = self.create_publisher(ArmJoint, outer.cfg.command_topic, 10)
                outer._subs.append(
                    self.create_subscription(ArmJoints, outer.cfg.feedback_topic, self.on_feedback, 10)
                )
                outer._subs.append(
                    self.create_subscription(ArmJoints, outer.cfg.raw_topic, self.on_raw, 10)
                )
                outer._subs.append(
                    self.create_subscription(Image, outer.cfg.color_topic, self.on_color, camera_qos)
                )
                outer._subs.append(
                    self.create_subscription(Image, outer.cfg.depth_topic, self.on_depth, camera_qos)
                )
                compressed_topic = outer.cfg.color_topic.rstrip("/") + "/compressed"
                outer._subs.append(
                    self.create_subscription(CompressedImage, compressed_topic, self.on_color_compressed, camera_qos)
                )

            def on_feedback(self, msg):
                values = {
                    1: int(msg.joint1), 2: int(msg.joint2), 3: int(msg.joint3),
                    4: int(msg.joint4), 5: int(msg.joint5), 6: int(msg.joint6),
                }
                outer.latest_feedback = values
                outer.feedback_stamp = time.monotonic()
                before = outer.safety.ready_joints()
                outer.safety.update_measured(values)
                outer._check_motion_feedback(values)
                after = outer.safety.ready_joints()
                if after != before and after:
                    print(f"[ARM] {outer.safety.status_line()}", flush=True)


            def on_raw(self, msg):
                outer.latest_raw = {
                    1: int(msg.joint1), 2: int(msg.joint2), 3: int(msg.joint3),
                    4: int(msg.joint4), 5: int(msg.joint5), 6: int(msg.joint6),
                }

            def on_color(self, msg):
                frame = outer._image_to_numpy(msg, self.bridge, "bgr8")
                if frame is not None:
                    outer.latest_color = frame
                    outer.color_stamp = time.monotonic()
                    if outer._last_camera_source != "raw":
                        outer._last_camera_source = "raw"
                        print(f"[CAMERA] receiving raw color {frame.shape}", flush=True)

            def on_color_compressed(self, msg):
                # Do not overwrite a healthy raw stream. Use compressed MJPEG
                # only if raw is absent/stale.
                if outer.latest_color is not None and time.monotonic() - outer.color_stamp < 0.5:
                    return
                frame = outer._compressed_to_numpy(msg)
                if frame is not None:
                    outer.latest_color = frame
                    outer.color_stamp = time.monotonic()
                    if outer._last_camera_source != "compressed":
                        outer._last_camera_source = "compressed"
                        print(f"[CAMERA] receiving compressed color {frame.shape}", flush=True)

            def on_depth(self, msg):
                frame = outer._image_to_numpy(msg, self.bridge, "passthrough")
                if frame is not None:
                    outer.latest_depth = frame
                    outer.depth_stamp = time.monotonic()

        self.node = NodeImpl()
        # Do not claim arm motion is ready until the STM32 subscription is visible.
        end = time.monotonic() + 5.0
        while time.monotonic() < end and self.node.pub.get_subscription_count() < 1:
            time.sleep(0.05)
        if self.node.pub.get_subscription_count() >= 1:
            print("[ARM] STM32 command subscriber matched", flush=True)
        else:
            print("[ARM] WARNING: /arm_joint has no matched STM32 subscriber yet", flush=True)
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, name="milo-ros", daemon=True)
        self.thread.start()
        self.available = True
        print("[ROS] runtime started", flush=True)


    def _check_motion_feedback(self, values: dict[int, int], now: float | None = None):
        now = time.monotonic() if now is None else now
        done = []
        for joint, pending in list(self._pending_motion.items()):
            value = values.get(joint)
            if value is None:
                continue
            before = pending.get("before")
            target = pending["target"]
            if not pending.get("movement_logged") and before is not None and value != before:
                print(f"[ARM ACK] J{joint} moving {before} -> {value} (target {target})", flush=True)
                pending["movement_logged"] = True
            if value == target or (
                pending.get("movement_logged") and abs(value - target) <= 1 and now >= pending["min_complete"]
            ):
                print(f"[ARM ACK] J{joint} reached {value} (target {target})", flush=True)
                done.append(joint)
            elif now >= pending["deadline"]:
                raw = self.latest_raw.get(joint)
                raw_text = f", raw={raw}" if raw is not None else ""
                print(f"[ARM WARN] J{joint} target {target}, feedback={value}{raw_text}", flush=True)
                done.append(joint)
        for joint in done:
            self._pending_motion.pop(joint, None)

    def arm_busy(self) -> bool:
        return bool(self._pending_motion)

    def _warn_encoding_once(self, encoding: str):
        if encoding not in self._decode_warnings:
            self._decode_warnings.add(encoding)
            print(f"[ROS] unsupported image encoding: {encoding}", flush=True)

    def _image_to_numpy(self, msg, bridge=None, desired_encoding=None):
        try:
            if bridge is not None:
                return bridge.imgmsg_to_cv2(msg, desired_encoding=desired_encoding or "passthrough")
            import numpy as np
            encoding = str(msg.encoding).lower()
            h, w, step = int(msg.height), int(msg.width), int(msg.step)
            raw = memoryview(msg.data)

            if encoding in {"bgr8", "rgb8", "8uc3"}:
                data = np.frombuffer(raw, dtype=np.uint8)
                row = data.reshape(h, step)[:, : w * 3]
                image = row.reshape(h, w, 3).copy()
                if encoding == "rgb8":
                    image = image[:, :, ::-1].copy()
                return image

            if encoding in {"bgra8", "rgba8", "8uc4"}:
                import cv2
                data = np.frombuffer(raw, dtype=np.uint8)
                row = data.reshape(h, step)[:, : w * 4]
                image = row.reshape(h, w, 4).copy()
                code = cv2.COLOR_BGRA2BGR if encoding != "rgba8" else cv2.COLOR_RGBA2BGR
                return cv2.cvtColor(image, code)

            if encoding in {"mono8", "8uc1"}:
                data = np.frombuffer(raw, dtype=np.uint8)
                return data.reshape(h, step)[:, :w].copy()

            if encoding in {"16uc1", "mono16"}:
                data16 = np.frombuffer(raw, dtype=np.uint16)
                return data16.reshape(h, step // 2)[:, :w].copy()

            if encoding in {"yuv422_yuy2", "yuyv", "yuy2"}:
                import cv2
                data = np.frombuffer(raw, dtype=np.uint8)
                row = data.reshape(h, step)[:, : w * 2]
                yuyv = row.reshape(h, w, 2)
                return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

            if encoding in {"uyvy", "yuv422"}:
                import cv2
                data = np.frombuffer(raw, dtype=np.uint8)
                row = data.reshape(h, step)[:, : w * 2]
                uyvy = row.reshape(h, w, 2)
                return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)

            self._warn_encoding_once(encoding)
        except Exception as exc:
            print(f"[ROS] image decode failed: {exc}", flush=True)
        return None

    def _compressed_to_numpy(self, msg):
        try:
            import cv2
            import numpy as np
            data = np.frombuffer(msg.data, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as exc:
            print(f"[ROS] compressed image decode failed: {exc}", flush=True)
            return None

    def camera_fresh(self, now=None):
        now = time.monotonic() if now is None else now
        return self.latest_color is not None and now - self.color_stamp <= self.cfg.camera_max_age

    def wait_for_color(self, timeout: float = 8.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.camera_fresh():
                return True
            time.sleep(0.05)
        return self.camera_fresh()

    def wait_for_feedback(self, timeout: float = 5.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if time.monotonic() - self.feedback_stamp <= self.cfg.feedback_max_age and self.latest_feedback:
                return True
            time.sleep(0.05)
        return time.monotonic() - self.feedback_stamp <= self.cfg.feedback_max_age and bool(self.latest_feedback)

    def command_joint(self, joint: int, raw_target: float) -> tuple[bool, str]:
        if not self.node:
            return False, "ROS node unavailable"
        if not self.camera_fresh():
            return False, "camera frame stale or missing"
        if self.arm_busy():
            return False, "waiting for previous arm motion feedback"
        result = self.safety.validate(joint, raw_target, self.cfg.runtime_ms)
        if not result.ok:
            return False, result.reason
        from arm_msgs.msg import ArmJoint
        msg = ArmJoint()
        msg.id = int(joint)
        msg.joint = int(result.target)
        msg.time = int(self.cfg.runtime_ms)
        if self.node.pub.get_subscription_count() < 1:
            return False, "STM32 /arm_joint subscriber not matched"

        before = self.safety.measured.get(joint)
        # Default ROS subscription is reliable. Send one absolute command and
        # wait for measured feedback before the next tracking interval. Repeating
        # every target three times in RC5 unnecessarily loaded the STM32 servo bus.
        self.node.pub.publish(msg)
        self.safety.note_command(joint, int(result.target))
        self._pending_motion[joint] = {
            "before": before,
            "target": int(result.target),
            "min_complete": time.monotonic() + self.cfg.runtime_ms / 1000.0,
            "deadline": time.monotonic() + max(1.5, self.cfg.runtime_ms / 1000.0 + 0.8),
            "movement_logged": False,
        }
        print(f"[ARM] command J{joint} -> {int(result.target)} ({self.cfg.runtime_ms} ms)", flush=True)
        return True, ""

    def command_aux_joint(
        self,
        joint: int,
        target: int,
        runtime_ms: int,
        limits: tuple[int, int] = (0, 180),
        max_speed_deg_s: float = 45.0,
    ) -> tuple[bool, str]:
        """Command a narrowly-scoped auxiliary posture joint outside face-follow safety."""
        if joint in (4, 6):
            return False, f"J{joint} is locked in MILO"
        if joint == 2 and not 90 <= target <= 165:
            return False, "J2 command outside 90..165 degrees"
        if not self.node:
            return False, "ROS node unavailable"
        if self.node.pub.get_subscription_count() < 1:
            return False, "STM32 /arm_joint subscriber not matched"
        if self.arm_busy():
            return False, "waiting for previous arm motion feedback"
        now = time.monotonic()
        if now - self.feedback_stamp > self.cfg.feedback_max_age:
            return False, f"J{joint} feedback stale"
        current = self.latest_feedback.get(joint)
        if current is None:
            return False, f"J{joint} feedback missing"
        if joint == 2 and target > 115 and target >= current:
            return False, "J2 cannot move down toward the box"
        lo, hi = limits
        current_raw = int(current)
        current_for_limits = current_raw
        soft = max(0, int(getattr(self.cfg, "feedback_soft_limit_deg", 0)))
        if current_raw < lo and current_raw >= lo - soft:
            current_for_limits = lo
        elif current_raw > hi and current_raw <= hi + soft:
            current_for_limits = hi
        if not lo <= current_for_limits <= hi:
            return False, f"J{joint} feedback {current_raw} outside command range {lo}..{hi}"
        target = int(round(target))
        if not lo <= target <= hi:
            return False, f"target {target} outside firmware range {lo}..{hi}"
        max_step = max(1, int(getattr(self.cfg, "startup_step_deg", 4)))
        if abs(target - current_for_limits) > max_step:
            return False, f"aux step exceeds {max_step}"
        seconds = max(0.001, runtime_ms / 1000.0)
        if abs(target - current_for_limits) / seconds > max_speed_deg_s + 1e-9:
            return False, "speed limit exceeded"

        from arm_msgs.msg import ArmJoint
        msg = ArmJoint()
        msg.id = int(joint)
        msg.joint = target
        msg.time = int(runtime_ms)
        self.node.pub.publish(msg)
        self._pending_motion[joint] = {
            "before": current_raw,
            "target": target,
            "min_complete": now + runtime_ms / 1000.0,
            "deadline": now + max(1.5, runtime_ms / 1000.0 + 0.8),
            "movement_logged": False,
        }
        print(f"[ARM] aux J{joint}: {current_raw} -> {target} ({runtime_ms} ms)", flush=True)
        return True, ""

    def command_startup_j4(self) -> tuple[bool, str]:
        """Move J4 to the verified upright startup angle without enabling tracking on J4."""
        joint = 4
        target = int(self.cfg.startup_j4_target)
        runtime_ms = int(self.cfg.startup_j4_runtime_ms)
        if not -30 <= target <= 115:
            return False, "J4 startup target outside -30..115"
        if not self.node:
            return False, "ROS node unavailable"
        if self.node.pub.get_subscription_count() < 1:
            return False, "STM32 /arm_joint subscriber not matched"
        if self.arm_busy():
            return False, "waiting for previous arm motion feedback"
        now = time.monotonic()
        if now - self.feedback_stamp > self.cfg.feedback_max_age:
            return False, "J4 feedback stale"
        current = self.latest_feedback.get(joint)
        if current is None:
            return False, "J4 feedback missing"
        current_raw = int(current)
        if abs(current_raw - target) <= 2:
            print(f"[ARM] startup J4 already at {current_raw}", flush=True)
            return True, ""
        if not -35 <= current_raw <= 125:
            return False, f"J4 feedback {current_raw} outside startup range -35..125"
        seconds = max(0.001, runtime_ms / 1000.0)
        if abs(target - current_raw) / seconds > self.cfg.startup_max_speed_deg_s + 1e-9:
            return False, "J4 startup speed limit exceeded"

        from arm_msgs.msg import ArmJoint
        msg = ArmJoint()
        msg.id = joint
        msg.joint = target
        msg.time = runtime_ms
        self.node.pub.publish(msg)
        self._pending_motion[joint] = {
            "before": current_raw,
            "target": target,
            "min_complete": now + runtime_ms / 1000.0,
            "deadline": now + max(1.5, runtime_ms / 1000.0 + 1.0),
            "movement_logged": False,
        }
        print(f"[ARM] startup J4: {current_raw} -> {target} ({runtime_ms} ms)", flush=True)
        return True, ""

    def close(self):
        try:
            if self.executor:
                self.executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        try:
            if self.node:
                self.node.destroy_node()
        except Exception:
            pass
        try:
            if self._rclpy and self._owns_context and self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass
