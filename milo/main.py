from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

from .audio import VoiceIO
from .config import Config
from .gaze import camera_to_gaze
from .intents import visual_intent
from .llm import LocalModel
from .memory import ObjectMemory
from .ros_runtime import RosRuntime
from .startup import step_toward
from .tracking import FaceFollower
from .ui import FaceUI
from .vision import VisionEngine


class Milo:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stop = threading.Event()
        self.ui = FaceUI(cfg.show_face)
        self.ros = RosRuntime(cfg)
        self.vision = VisionEngine(cfg)
        self.model = LocalModel(cfg.llm_url, str(cfg.llm_model), cfg.llm_max_tokens, cfg.llm_timeout)
        self.object_memory = ObjectMemory(cfg.object_memory_path)
        self.voice = VoiceIO(cfg, self.ui.set_speaking)
        self.follower = FaceFollower(cfg)
        self.last_face_motion = 0.0
        self.last_motion_warning = ""
        self.vision_thread = None
        self.proactive_thread = None
        self._last_face_log = 0.0
        self._last_no_face_log = 0.0
        self._last_face_seen = 0.0
        self._last_user_speech = time.monotonic()
        self._last_proactive_speech = 0.0
        self._voice_lock = threading.Lock()

    def shutdown(self, *_):
        self.stop.set()

    def start(self):
        self.ui.start()
        self.ros.start()
        self._startup_lift()

        camera_ready = self.ros.wait_for_color(8.0)
        if camera_ready:
            shape = getattr(self.ros.latest_color, "shape", None)
            print(f"[READY] application receives DaBai color frames {shape}", flush=True)
        else:
            print("[DEGRADED] application receives no DaBai color frames; voice will continue, tracking disabled", flush=True)

        self.vision_thread = threading.Thread(target=self._vision_loop, name="milo-vision", daemon=True)
        self.vision_thread.start()
        self.proactive_thread = threading.Thread(target=self._proactive_loop, name="milo-proactive", daemon=True)
        self.proactive_thread.start()
        print("[READY] MILO Clean RC13 runtime", flush=True)
        if camera_ready:
            print("[READY] voice + DaBai vision + cat face + feedback-gated J1 pan + J3 tilt face following", flush=True)
        else:
            print("[READY] voice + cat face; vision/arm tracking waiting for camera", flush=True)
        self._voice_loop()

    def _startup_lift(self):
        if not self.cfg.startup_lift_enabled:
            print("[STARTUP LIFT] disabled", flush=True)
            return
        if not self.ros.wait_for_feedback(6.0):
            print("[STARTUP LIFT] skipped: no fresh arm feedback", flush=True)
            return

        print("[STARTUP LIFT] gently aligning J4/J3/J2 before face following", flush=True)
        deadline = time.monotonic() + self.cfg.startup_timeout_sec
        ok, reason = self.ros.command_startup_j4()
        if not ok:
            print(f"[STARTUP LIFT] J4 screen home blocked: {reason}", flush=True)
        while self.ros.arm_busy() and not self.stop.wait(0.05):
            pass

        plan = ((3, self.cfg.startup_j3_target, "screen lift"), (2, self.cfg.startup_j2_target, "downward-box guard"))
        for joint, target, label in plan:
            joint_completed = False
            last_attempt = None
            repeated_attempts = 0
            while not self.stop.is_set() and time.monotonic() < deadline:
                if not self.ros.wait_for_feedback(2.0):
                    print(f"[STARTUP LIFT] J{joint} stopped: feedback stale", flush=True)
                    break
                current = int(self.ros.latest_feedback.get(joint, target))

                if joint == 2:
                    # Bring the shoulder to the verified startup angle from
                    # either side. From a folded pose this lifts away from the
                    # box; from an over-upright pose it returns to the working
                    # camera geometry.
                    if abs(current - target) <= 2:
                        print(f"[STARTUP LIFT] J2 already safe at {current}", flush=True)
                        joint_completed = True
                        break
                    next_target = step_toward(current, target, self.cfg.startup_step_deg)
                    if next_target is None:
                        print(f"[STARTUP LIFT] J2 already safe at {current}", flush=True)
                        joint_completed = True
                        break
                else:
                    if abs(current - target) <= 2:
                        print(f"[STARTUP LIFT] J3 already at startup angle {current}", flush=True)
                        joint_completed = True
                        break
                    next_target = step_toward(current, target, self.cfg.startup_step_deg)
                    if next_target is None:
                        print(f"[STARTUP LIFT] J3 already at startup angle {current}", flush=True)
                        joint_completed = True
                        break

                if next_target == last_attempt:
                    repeated_attempts += 1
                    if repeated_attempts >= 2:
                        print(f"[STARTUP LIFT] J{joint} stopped: no feedback progress toward {target}", flush=True)
                        break
                else:
                    last_attempt = next_target
                    repeated_attempts = 0

                ok, reason = self.ros.command_aux_joint(
                    joint,
                    next_target,
                    self.cfg.startup_runtime_ms,
                    max_speed_deg_s=self.cfg.startup_max_speed_deg_s,
                )
                if not ok:
                    print(f"[STARTUP LIFT] J{joint} {label} blocked: {reason}", flush=True)
                    break
                while self.ros.arm_busy() and not self.stop.wait(0.05):
                    pass
            else:
                if time.monotonic() >= deadline:
                    print("[STARTUP LIFT] stopped by timeout", flush=True)
                    break
            if joint == 3 and not joint_completed:
                print("[STARTUP LIFT] J2 skipped because J3 did not reach its safe startup angle", flush=True)
                break
        print("[STARTUP LIFT] done; face following may start", flush=True)

    def _vision_loop(self):
        last_error = ""
        while not self.stop.wait(0.04):
            try:
                frame = self.ros.latest_color
                fresh = frame is not None and time.monotonic() - self.ros.color_stamp <= self.cfg.camera_max_age
                if fresh:
                    obs = self.vision.detect_face(frame)
                    now = time.monotonic()
                    if obs.center is not None:
                        cx, cy = obs.center
                        self._last_face_seen = now
                        self.ui.set_gaze(*camera_to_gaze(cx, cy, self.cfg.cat_gaze_x_sign, self.cfg.cat_gaze_y_sign))
                        if now - self._last_face_log >= 1.0:
                            print(
                                f"[FACE] found x={cx:.3f} y={cy:.3f} score={obs.score:.3f} roll={obs.roll_deg:.1f}",
                                flush=True,
                            )
                            self._last_face_log = now
                        if now - self.last_face_motion >= self.cfg.face_interval:
                            self.last_face_motion = now
                            hold = self.follower.motion_hold_reason(obs.box)
                            if hold:
                                if hold != self.last_motion_warning:
                                    print(f"[MOTION HOLD] {hold}", flush=True)
                                    self.last_motion_warning = hold
                            else:
                                self._follow_face(cx, cy, obs.roll_deg)
                    else:
                        if now - self._last_face_seen >= 0.6:
                            self.ui.set_gaze(0.0, 0.0)
                        if now - self._last_no_face_log >= 2.0:
                            print("[FACE] no face", flush=True)
                            self._last_no_face_log = now
                else:
                    if time.monotonic() - self._last_face_seen >= 0.6:
                        self.ui.set_gaze(0.0, 0.0)
                    time.sleep(0.05)
                last_error = ""
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message != last_error:
                    print(f"[VISION] loop error (continuing): {message}", flush=True)
                    last_error = message
                self.stop.wait(0.5)

    def _follow_face(self, cx: float, cy: float, roll_deg: float):
        ready = self.ros.safety.ready_joints()
        positions = self.ros.safety.trusted_positions()
        if not ready:
            reason = self.ros.safety.status_line()
            if reason != self.last_motion_warning:
                print(f"[MOTION BLOCKED] {reason}", flush=True)
                self.last_motion_warning = reason
            return

        plan = self.follower.plan(cx, cy, roll_deg, positions, ready, self.ros.safety.local_limits)
        if plan is None:
            self.last_motion_warning = ""
            return

        joint, target = plan
        ok, reason = self.ros.command_joint(joint, target)
        if ok:
            if self.last_motion_warning:
                print("[MOTION READY] face following active", flush=True)
            self.last_motion_warning = ""
        elif reason != self.last_motion_warning:
            print(f"[MOTION BLOCKED] {reason}", flush=True)
            self.last_motion_warning = reason

    def _voice_loop(self):
        while not self.stop.is_set():
            text = self.voice.listen(self.stop)
            if text:
                self._last_user_speech = time.monotonic()
                print(f"YOU: {text}", flush=True)
                intent, item = visual_intent(text)
                try:
                    if intent == "object":
                        answer = self._locate_object(text, item)
                    elif intent == "scene":
                        fresh = self.ros.camera_fresh()
                        jpeg = self.vision.jpeg(self.ros.latest_color) if fresh else None
                        answer = self.model.describe(text, jpeg) if jpeg else "I cannot see a fresh camera frame right now."
                    else:
                        answer = self.model.chat(text)
                except Exception as exc:
                    print(f"[LLM] request failed: {exc}", flush=True)
                    answer = "My local language model is temporarily unavailable."
                answer = str(answer or "").strip()
                if not answer:
                    answer = "I heard you, but I did not get a usable response from my local model."
                print(f"MILO: {answer}", flush=True)
                with self._voice_lock:
                    self.voice.speak(answer)
            elif self.voice.is_speaking.is_set():
                self.stop.wait(0.1)

    def _say(self, text: str) -> None:
        text = str(text or "").strip()
        if not text or self.stop.is_set():
            return
        with self._voice_lock:
            if not self.stop.is_set():
                print(f"MILO: {text}", flush=True)
                self.voice.speak(text)

    def _proactive_loop(self):
        greeted = False
        while not self.stop.wait(1.0):
            now = time.monotonic()
            if not greeted and self.ros.camera_fresh() and (now - self._last_face_seen <= 10.0):
                try:
                    greeting = self.model.proactive_reply(
                        "Start the conversation as MILO. Say hello in English, introduce yourself as my assistant, and ask what you can help with."
                    )
                except Exception as exc:
                    print(f"[LLM] greeting failed: {exc}", flush=True)
                    greeting = "Hi, I'm MILO, your assistant. What can I help you with?"
                self._say(greeting)
                greeted = True
                self._last_proactive_speech = time.monotonic()
                continue

            if not greeted and self.ros.camera_fresh() and now - self._last_proactive_speech > 12:
                try:
                    greeting = self.model.proactive_reply(
                        "Start the conversation as MILO. Say hello in English, introduce yourself as my assistant, and ask what you can help with."
                    )
                except Exception as exc:
                    print(f"[LLM] greeting failed: {exc}", flush=True)
                    greeting = "Hi, I'm MILO, your assistant. What can I help you with?"
                self._say(greeting)
                greeted = True
                self._last_proactive_speech = time.monotonic()
                continue

            if now - self._last_user_speech < 60 or now - self._last_proactive_speech < 150:
                continue
            if self.voice.is_speaking.is_set() or not self.ros.camera_fresh():
                continue
            if now - self._last_face_seen > 30:
                continue
            jpeg = self.vision.jpeg(self.ros.latest_color)
            if not jpeg:
                continue
            try:
                observation = self.model.observe_person(jpeg)
            except Exception as exc:
                print(f"[VLM] proactive observation failed: {exc}", flush=True)
                continue
            if not observation.get("person_visible") or observation.get("confidence", 0.0) < 0.55:
                continue
            instruction = ""
            if observation.get("phone_visible"):
                instruction = "I can see the user sitting with a phone. Say a short friendly nudge in English: why are you on your phone, go work."
            elif observation.get("idle"):
                instruction = "I can see the user sitting and not doing much. Say a short friendly nudge in English that they should stretch or move a little."
            if not instruction:
                continue
            try:
                answer = self.model.proactive_reply(instruction)
            except Exception as exc:
                print(f"[LLM] proactive reply failed: {exc}", flush=True)
                continue
            self._say(answer)
            self._last_proactive_speech = time.monotonic()



    def _locate_object(self, request: str, item: str) -> str:
        current = None
        if self.ros.camera_fresh():
            jpeg = self.vision.jpeg(self.ros.latest_color)
            if jpeg:
                try:
                    current = self.model.locate_object(request, item, jpeg)
                except Exception as exc:
                    print(f"[VLM] object search failed: {exc}", flush=True)
                    current = {"object": item, "visible": False, "location": "", "reliable": False}
        label = item or (current or {}).get("object") or "that item"
        if current and current["visible"]:
            location = current["location"]
            try:
                self.object_memory.remember(label, location)
            except OSError as exc:
                print(f"[MEMORY] could not save sighting: {exc}", flush=True)
            answer = f"I can see your {label} {location}."
        else:
            previous = self.object_memory.recall(label)
            if previous:
                answer = (
                    f"I can't see your {label} in the current view. "
                    f"The last time I saw it, it was {previous['location']} "
                    f"at {previous.get('seen_at', 'an earlier time')}. It may have moved since then."
                )
            elif current and not current["reliable"]:
                answer = f"I can't reliably identify your {label} in this view. Could you show me another area?"
            else:
                answer = f"I can't see your {label} in this view, and I don't have a previous sighting. Could you show me another area?"
        self.model.remember_exchange(request, answer)
        return answer

    def close(self):
        self.stop.set()
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2)
        if self.proactive_thread and self.proactive_thread.is_alive():
            self.proactive_thread.join(timeout=2)
        self.vision.close()
        self.ros.close()
        self.ui.stop()


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = Config.load(root)
    milo = Milo(cfg)
    signal.signal(signal.SIGINT, milo.shutdown)
    signal.signal(signal.SIGTERM, milo.shutdown)
    try:
        milo.start()
    finally:
        milo.close()


if __name__ == "__main__":
    main()
