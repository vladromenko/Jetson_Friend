import argparse
import fcntl
import os
import queue
import re
import signal
import threading
import time

from ai import HughAI
from behavior import Behavior
from emotion import EmotionManager
from face import Face
from identity import IdentityManager
from intents import wants_visual
from memory import Memory
from memory_policy import candidate, private_turn
from speech import Speech
from telemetry import TurnTiming
from vision import Vision
from robotics.manager import RobotManager


def extract_name(text):
    """Extract a name only from an explicit naming statement."""
    patterns = [
        "\\bmy name is\\s+([A-Za-z][A-Za-z\\-']{1,30})\\b",
        "\\bcall me\\s+([A-Za-z][A-Za-z\\-']{1,30})\\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text or ""), flags=re.I)
        if match:
            return match.group(1).strip().title()
    return None


def asks_assistant_identity(text):
    return bool(
        re.search(
            "\\b(who are you|what is your name|what's your name|tell me who you are|tell me your name|who am i talking to)\\b",
            str(text or ""),
            flags=re.I,
        )
    )


def confirms_name(text):
    cleaned = re.sub("[^a-z']+", " ", str(text or "").lower()).strip()
    return bool(
        re.fullmatch(
            "(?:yes|yeah|yep|correct|right|that's right|that is right|yes that's right|yes that is right)",
            cleaned,
        )
    )


def rejects_name(text):
    cleaned = re.sub("[^a-z']+", " ", str(text or "").lower()).strip()
    return bool(
        re.fullmatch(
            "(?:no|nope|wrong|that's wrong|that is wrong|no that's wrong|no that is wrong)",
            cleaned,
        )
    )


def looks_visual(text):
    return wants_visual(text)


def detect_explicit_emotion(text):
    """
    Detect only direct first-person emotion statements.

    Facial emotion remains a weak visual cue. Explicit speech
    is treated as the higher-confidence source.
    """
    lowered = str(text or "").strip().lower()
    if not lowered:
        return None
    negated = re.search(
        "\\b(i(?:'m| am)|i feel|i'm feeling|i am feeling)\\s+(?:really\\s+|very\\s+|pretty\\s+|a bit\\s+|a little\\s+)?not\\s+",
        lowered,
        flags=re.I,
    )
    if negated:
        return None
    emotion_words = {
        "happy": "happy",
        "glad": "happy",
        "great": "happy",
        "good": "happy",
        "excited": "excited",
        "sad": "sad",
        "down": "sad",
        "upset": "upset",
        "angry": "angry",
        "mad": "angry",
        "furious": "angry",
        "nervous": "anxious",
        "anxious": "anxious",
        "worried": "anxious",
        "stressed": "stressed",
        "afraid": "afraid",
        "scared": "afraid",
        "frustrated": "frustrated",
        "lonely": "lonely",
        "calm": "calm",
        "relaxed": "calm",
    }
    words = "|".join(sorted(emotion_words, key=len, reverse=True))
    patterns = [
        f"\\b(?:i feel|i'm feeling|i am feeling)\\s+(?:really\\s+|very\\s+|pretty\\s+|a bit\\s+|a little\\s+)?({words})\\b",
        f"\\b(?:i'm|i am)\\s+(?:really\\s+|very\\s+|pretty\\s+|a bit\\s+|a little\\s+)?({words})\\b",
    ]
    detected = None
    for pattern in patterns:
        if detected is None:
            match = re.search(pattern, lowered, flags=re.I)
            if match:
                word = match.group(1).lower()
                detected = emotion_words.get(word)
    return detected


def asks_memory_summary(text):
    return bool(
        re.search(
            "\\b(what do you remember about me|what do you know about me|tell me what you remember about me)\\b",
            text,
            flags=re.I,
        )
    )


def asks_forget_last(text):
    return bool(
        re.search(
            "\\b(forget that|don't remember that|do not remember that)\\b",
            text,
            flags=re.I,
        )
    )


def asks_forget_person(text):
    return bool(
        re.search(
            "\\b(forget everything about me|delete everything about me|delete my profile|forget me completely)\\b",
            text,
            flags=re.I,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    for option in (
        "debug",
        "no-face",
        "no-vision",
        "no-mic",
        "no-identity",
        "no-emotion",
    ):
        parser.add_argument("--" + option, action="store_true")
    args = parser.parse_args()
    runtime_root = os.getenv(
        "JETSON_FRIEND_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    os.makedirs(os.path.join(runtime_root, "data"), exist_ok=True)
    instance_lock = open(os.path.join(runtime_root, "data", "milo.lock"), "a")
    try:
        fcntl.flock(instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        instance_lock.close()
        print("MILO is already running.", flush=True)
        return
    memory = Memory()
    behavior = Behavior(memory)
    face, speech, vision = Face(), Speech(), Vision()
    identity = IdentityManager(memory)
    # Facial emotion inference is opt-in; explicit words require no neural model.
    emotion = (
        EmotionManager(memory)
        if not args.no_emotion and os.getenv("MILO_VISUAL_EMOTION", "0") == "1"
        else None
    )
    ai = HughAI()
    speech.warmup()
    stop, busy, listening = threading.Event(), threading.Event(), threading.Event()
    messages = queue.Queue(maxsize=8)
    lock = threading.RLock()
    state = {
        "person_id": None,
        "name": None,
        "epoch": 0,
        "track_id": None,
        "pending_name": None,
        "scene": {},
        "manual_until": 0.0,
    }
    threads = []
    last_seen_write = {}
    observed_objects = set()

    def active():
        with lock:
            return state["person_id"], state["name"], state["epoch"]

    def select(person_id, name=None, track_id=None):
        if person_id and not memory.get_person(person_id):
            person_id, name = None, None
        changed = False
        with lock:
            if (person_id, track_id) != (state["person_id"], state["track_id"]):
                changed = True
                state["epoch"] += 1
                state["pending_name"] = None
                state["manual_until"] = 0.0
                if emotion:
                    emotion.reset_person()
            state.update(person_id=person_id, name=name, track_id=track_id)
        if changed:
            if speech.is_speaking.is_set():
                speech.stop_speaking()
            memory.set_current_person(person_id)

    robot = None

    def shutdown(*_):
        # Stop new arm commands immediately, before waiting on audio/vision shutdown.
        if robot is not None:
            robot.stop_request.set()
            robot.armed = False
        stop.set()
        speech.stop_speaking()
        vision.stop()
        face.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    def start_thread(target, name, args=(), daemon=True):
        thread = threading.Thread(target=target, name=name, args=args, daemon=daemon)
        thread.start()
        threads.append(thread)

    def enqueue(text, source, initial=None, timing=None):
        current = active()
        # If the audience changed while listening, treat this as an anonymous turn.
        audience = (
            current
            if initial is None or initial == current
            else (None, None, current[2])
        )
        try:
            messages.put_nowait(
                (text, source, audience, timing or TurnTiming(source=source))
            )
        except queue.Full:
            print("Input queue full; discarded stale input.", flush=True)

    def keyboard():
        while not stop.is_set():
            try:
                text = input("> " if args.debug else "")
            except (EOFError, KeyboardInterrupt):
                return
            if text.strip():
                enqueue(text.strip(), "keyboard")

    def microphone():
        while not stop.is_set():
            if busy.is_set() or speech.is_speaking.is_set():
                stop.wait(0.05)
                continue
            initial = active()

            def mic_state(value):
                if value == "listening":
                    listening.set()
                face.set_state(value)

            try:
                text = speech.listen_once(mic_state, stop_event=stop)
                if text and not stop.is_set() and not busy.is_set():
                    timing = TurnTiming(
                        start=getattr(speech, "last_speech_end", None),
                        source="microphone",
                    )
                    timing.mark("stt_complete")
                    enqueue(text, "microphone", initial, timing)
            except Exception as exc:
                print(f"Microphone recovering: {exc}", flush=True)
                stop.wait(1.0)
            finally:
                listening.clear()

    def awareness():
        nonlocal observed_objects
        while not stop.is_set():
            try:
                scene = vision.get_scene() if not args.no_vision else {}
                fresh = time.time() - scene.get("updated", 0) < 2
                tracks = scene.get("face_tracks", []) if fresh else []
                people = []
                if not args.no_identity and identity.available:
                    for track in tracks:
                        result = identity.recognize_from_vision(
                            vision, track_id=track["track_id"]
                        )
                        people.append(
                            {**result.as_dict(), "track_id": track["track_id"]}
                        )
                else:
                    people = [
                        {"person_id": None, "track_id": t["track_id"]} for t in tracks
                    ]
                with lock:
                    manual = state["manual_until"] > time.monotonic()
                    manual_track = state["track_id"]
                if not args.no_vision:
                    if len(people) == 1 and people[0].get("person_id"):
                        p = people[0]
                        select(p["person_id"], p["name"], p["track_id"])
                    elif not (
                        manual
                        and len(people) == 1
                        and people[0]["track_id"] == manual_track
                    ):
                        select(
                            None,
                            track_id=tracks[0]["track_id"]
                            if len(tracks) == 1
                            else None,
                        )
                elif not manual:
                    select(None)
                scene["people"] = people
                behavior.update(scene, busy=busy.is_set(), listening=listening.is_set())
                with lock:
                    state["scene"] = scene
                for p in people:
                    pid = p.get("person_id")
                    if pid and time.time() - last_seen_write.get(pid, 0) >= 30:
                        memory.touch_person(pid)
                        last_seen_write[pid] = time.time()
                # Store class-level appearance transitions, never infer ownership/use.
                objects = set(scene.get("objects", [])) if fresh else observed_objects
                for label in objects - observed_objects:
                    memory.see_object(label)
                observed_objects = objects
            except Exception as exc:
                select(None)
                with lock:
                    state["scene"] = {}
                print(f"Awareness recovering: {exc}", flush=True)
            stop.wait(0.25)

    def say(text, mood="neutral", timing=None, audience=None):
        if stop.is_set() or (audience and audience != active()):
            return
        face.set_state(mood)
        speech.speak(text, face.set_state, timing=timing)
        behavior.note_speech()
        face.set_state("neutral")


    robot = RobotManager()
    robot.bind_speech(lambda message, mood="neutral": say(message, mood))

    def stream(text, pid, name, context, timing, audience):
        phrases = queue.Queue(maxsize=4)
        cancel = threading.Event()

        def produce():
            try:
                for phrase in ai.stream_reply(
                    text,
                    person_id=pid,
                    person_name=name,
                    memory_context=context,
                    timing=timing,
                ):
                    while not cancel.is_set():
                        try:
                            phrases.put(phrase, timeout=0.2)
                            break
                        except queue.Full:
                            pass
                    if cancel.is_set():
                        break
            except Exception as exc:
                print(f"Streaming failed: {exc}", flush=True)
            finally:
                while not cancel.is_set():
                    try:
                        phrases.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        pass

        worker = threading.Thread(target=produce, name="llm-stream", daemon=True)
        worker.start()
        spoken = False
        try:
            while not stop.is_set():
                try:
                    phrase = phrases.get(timeout=0.2)
                except queue.Empty:
                    continue
                if phrase is None:
                    break
                if audience != active():
                    break
                say(phrase, timing=timing, audience=audience)
                spoken = True
        finally:
            cancel.set()
            worker.join(timeout=ai.timeout + 1)
        if not spoken and audience == active():
            say(
                "I couldn't finish that reply. Try me again.",
                "concerned",
                timing,
                audience,
            )

    if not args.no_face:
        start_thread(face.run, "face")
    if not args.no_vision:
        start_thread(vision.run, "vision", (face,))
    start_thread(awareness, "awareness")
    start_thread(keyboard, "keyboard")
    if not args.no_mic:
        start_thread(microphone, "microphone")
    print("MILO ready.", flush=True)
    try:
        while not stop.is_set():
            try:
                text, source, audience, timing = messages.get(timeout=0.25)
            except queue.Empty:
                with lock:
                    scene = dict(state["scene"])
                event = behavior.update(
                    scene, busy=busy.is_set(), listening=listening.is_set()
                )
                if event and event["person_id"] == active()[0]:
                    busy.set()
                    try:
                        behavior.acknowledge(event)
                        say(event["text"], "happy", audience=active())
                    finally:
                        busy.clear()
                continue
            busy.set()
            try:
                if audience != active():
                    audience = (None, None, active()[2])
                pid, name, _ = audience
                guarded_audience = active()
                suppress = private_turn(text)
                with lock:
                    pending = state["pending_name"]
                    track_id = state["track_id"]
                detected_name = extract_name(text)
                robot_result = robot.handle_voice(text)
                if robot_result.handled:
                    if robot_result.text:
                        say(robot_result.text, robot_result.mood, timing)
                elif asks_assistant_identity(text):
                    say("I'm MILO, your local AI companion.", "happy", timing)
                elif text.strip().lower().rstrip("?!.,") in {"milo", "hey milo"}:
                    say("Yeah, I'm listening.", timing=timing)
                elif asks_forget_person(text):
                    if not pid:
                        say(
                            "I need to recognize you before I can delete your profile.",
                            timing=timing,
                        )
                    else:
                        memory.forget_person(pid)
                        identity.clear_current_identity()
                        ai.clear_history(pid)
                        behavior.forget_person(pid)
                        if emotion:
                            emotion.reset_person(pid)
                        select(None)
                        say(
                            "Okay. I deleted your profile, face references, and memories from the active database.",
                            timing=timing,
                        )
                elif asks_forget_last(text):
                    forgotten = memory.forget_last(person_id=pid)
                    ai.clear_history(pid) if pid else None
                    say(
                        "Okay, I've forgotten that."
                        if forgotten
                        else "I don't have a recent memory to forget.",
                        timing=timing,
                    )
                elif asks_memory_summary(text):
                    # Direct retrieval avoids model invention and unnecessary latency.
                    say(
                        memory.describe_person(limit=5, person_id=pid),
                        timing=timing,
                        audience=guarded_audience,
                    )
                elif not suppress and re.match(r"(?:please )?remember\b", text, re.I):
                    item = candidate(text)
                    if not pid:
                        say(
                            "I need to recognize you before I can save a personal memory.",
                            timing=timing,
                        )
                    elif item and memory.remember(
                        **item, person_id=pid, confidence=0.95, source="explicit_text"
                    ):
                        say(
                            "Okay, I've saved that.",
                            timing=timing,
                            audience=guarded_audience,
                        )
                    else:
                        say("I'll leave that out of long-term memory.", timing=timing)
                elif detected_name and not suppress:
                    with lock:
                        state["pending_name"] = (
                            detected_name,
                            time.monotonic(),
                            state["epoch"],
                        )
                    say(f"I heard {detected_name}. Is that right?", "curious", timing)
                elif (
                    pending
                    and confirms_name(text)
                    and time.monotonic() - pending[1] <= 30
                    and pending[2] == active()[2]
                ):
                    if not args.no_vision and track_id is None:
                        say(
                            "Let's do that when I can see just you clearly.",
                            timing=timing,
                        )
                    else:
                        # A name is a label, never a credential for an existing profile.
                        new_pid = memory.create_person(pending[0], make_current=False)
                        select(new_pid, pending[0], track_id)
                        with lock:
                            state["manual_until"] = time.monotonic() + 60
                        memory.remember(
                            f"My name is {pending[0]}.",
                            kind="identity",
                            person_id=new_pid,
                            source="onboarding",
                            importance=1,
                        )
                        enrollment = {"ok": False}
                        if (
                            identity.available
                            and not args.no_identity
                            and not args.no_vision
                        ):
                            enrollment = identity.enroll_person_from_vision(
                                pending[0], vision, person_id=new_pid, timeout=6
                            )
                        say(
                            f"Nice to meet you, {pending[0]}. "
                            + (
                                "I saved several face references."
                                if enrollment["ok"]
                                else "Your profile is saved; I still need clearer face samples for recognition."
                            ),
                            "happy",
                            timing,
                        )
                    with lock:
                        state["pending_name"] = None
                elif pending and rejects_name(text):
                    with lock:
                        state["pending_name"] = None
                    say("Okay. Tell me again using 'My name is ...'.", timing=timing)
                else:
                    if suppress and pid:
                        ai.clear_history(pid)
                    context = "" if suppress else memory.context(text, person_id=pid)
                    timing.mark("memory_complete")
                    explicit_emotion = detect_explicit_emotion(text)
                    if pid and explicit_emotion and not suppress:
                        memory.set_temporary_state(
                            "emotion",
                            {"emotion": explicit_emotion, "source": "explicit_speech"},
                            person_id=pid,
                            ttl_seconds=600,
                        )
                    if looks_visual(text) and not args.no_vision:
                        vision.request_visual_attention(3)
                        reply = ai.ask_visual(
                            text,
                            vision.snapshot_jpeg(),
                            person_id=None if suppress else pid,
                            person_name=name,
                            memory_context=context,
                            timing=timing,
                        )
                        say(
                            reply["text"],
                            reply.get("emotion"),
                            timing,
                            guarded_audience,
                        )
                    elif os.getenv("LLM_STREAM", "0") == "1":
                        stream(
                            text,
                            None if suppress else pid,
                            name,
                            context,
                            timing,
                            guarded_audience,
                        )
                    else:
                        reply = ai.ask(
                            text,
                            scene=vision.get_scene(),
                            person_id=None if suppress else pid,
                            person_name=name,
                            memory_context=context,
                            timing=timing,
                        )
                        say(
                            reply["text"],
                            reply.get("emotion"),
                            timing,
                            guarded_audience,
                        )
                    item = candidate(text)
                    if item and pid and guarded_audience == active():
                        memory.remember(
                            **item,
                            person_id=pid,
                            confidence=0.95,
                            source="explicit_text",
                        )
            except Exception as exc:
                print(f"Conversation recovering: {exc}", flush=True)
                face.set_state("concerned")
            finally:
                timing.finish()
                busy.clear()
    finally:
        shutdown()
        try:
            robot.close()
        except Exception:
            pass
        ai.close()
        for thread in threads:
            thread.join(timeout=1)
        instance_lock.close()
        print("MILO stopped.", flush=True)


if __name__ == "__main__":
    main()
