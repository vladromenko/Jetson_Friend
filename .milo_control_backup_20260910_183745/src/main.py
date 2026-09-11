import argparse
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
from memory import Memory
from speech import Speech
from vision import Vision


def speak_reply(
    speech,
    face,
    text,
    emotion="neutral",
):
    emotion = emotion or "neutral"

    face.set_state(
        emotion
    )

    time.sleep(
        0.10
    )

    speech.speak(
        text,
        face.set_state,
    )

    face.set_state(
        emotion
    )

    time.sleep(
        0.20
    )

    face.set_state(
        "neutral"
    )


def extract_name(text):
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\-']{1,30})",
        r"\bcall me\s+([A-Za-z][A-Za-z\-']{1,30})",
        r"\bi am\s+([A-Za-z][A-Za-z\-']{1,30})",
        r"\bi'm\s+([A-Za-z][A-Za-z\-']{1,30})",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.I,
        )

        if match:
            return match.group(
                1
            ).strip().title()

    words = re.findall(
        r"[A-Za-z][A-Za-z\-']+",
        text,
    )

    if (
        len(words) == 1
        and 1 < len(words[0]) <= 30
    ):
        return words[0].title()

    return None


def looks_visual(text):
    return bool(
        re.search(
            (
                r"\b("
                r"see|seeing|look|looking|camera|face|"
                r"wearing|holding|room|around me|surroundings|"
                r"describe me|describe what|describe the|"
                r"in front of you|what is this|what's this|"
                r"what am i holding|what do i have|who is here"
                r")\b"
            ),
            text,
            flags=re.I,
        )
    )


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
        (
            r"\b(i(?:'m| am)|i feel|i'm feeling|i am feeling)\s+"
            r"(?:really\s+|very\s+|pretty\s+|a bit\s+|a little\s+)?"
            r"not\s+"
        ),
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

    words = "|".join(
        sorted(
            emotion_words,
            key=len,
            reverse=True,
        )
    )

    patterns = [
        (
            r"\b(?:i feel|i'm feeling|i am feeling)\s+"
            r"(?:really\s+|very\s+|pretty\s+|a bit\s+|a little\s+)?"
            rf"({words})\b"
        ),
        (
            r"\b(?:i'm|i am)\s+"
            r"(?:really\s+|very\s+|pretty\s+|a bit\s+|a little\s+)?"
            rf"({words})\b"
        ),
    ]

    detected = None

    for pattern in patterns:
        if detected is None:
            match = re.search(
                pattern,
                lowered,
                flags=re.I,
            )

            if match:
                word = match.group(1).lower()
                detected = emotion_words.get(word)

    return detected


def asks_memory_summary(text):
    return bool(
        re.search(
            (
                r"\b("
                r"what do you remember about me|"
                r"what do you know about me|"
                r"tell me what you remember about me"
                r")\b"
            ),
            text,
            flags=re.I,
        )
    )


def asks_forget_last(text):
    return bool(
        re.search(
            (
                r"\b("
                r"forget that|"
                r"don't remember that|"
                r"do not remember that"
                r")\b"
            ),
            text,
            flags=re.I,
        )
    )


def asks_forget_person(text):
    return bool(
        re.search(
            (
                r"\b("
                r"forget everything about me|"
                r"delete everything about me|"
                r"delete my profile|"
                r"forget me completely"
                r")\b"
            ),
            text,
            flags=re.I,
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--no-face",
        action="store_true",
    )

    parser.add_argument(
        "--no-vision",
        action="store_true",
    )

    parser.add_argument(
        "--no-mic",
        action="store_true",
    )

    parser.add_argument(
        "--no-identity",
        action="store_true",
    )

    parser.add_argument(
        "--no-emotion",
        action="store_true",
    )

    args = parser.parse_args()

    memory = Memory()

    behavior = Behavior(
        memory
    )

    face = Face()

    speech = Speech()

    vision = Vision()

    identity = IdentityManager(
        memory
    )

    emotion = EmotionManager(
        memory
    )

    assistant_name = "MILO"

    print(
        f"Starting {assistant_name}...",
        flush=True,
    )

    ai = HughAI()

    stop = threading.Event()

    busy = threading.Event()

    messages = queue.Queue()

    threads = []

    state_lock = threading.RLock()

    state = {
        "awaiting_name": False,
        "current_person_id": (
            memory.get_current_person_id()
        ),
        "current_name": (
            memory.get_profile(
                "name"
            )
        ),
        "identity_status": "LEGACY",
        "unknown_since": None,
        "last_identity_change": 0.0,
        "last_emotion": None,
    }

    last_object_write = {}

    last_identity_check = 0.0

    last_reinforcement = {}

    def debug_print(
        *items,
    ):
        if args.debug:
            print(
                *items,
                flush=True,
            )

    def set_active_person(
        person_id,
        name,
        source="identity",
    ):
        if not person_id:
            return False

        changed = False

        with state_lock:
            if (
                state[
                    "current_person_id"
                ]
                != person_id
            ):
                changed = True

            state[
                "current_person_id"
            ] = person_id

            state[
                "current_name"
            ] = name

            state[
                "identity_status"
            ] = source

            state[
                "unknown_since"
            ] = None

            if changed:
                state[
                    "last_identity_change"
                ] = time.time()

        memory.set_current_person(
            person_id
        )

        memory.touch_person(
            person_id
        )

        if changed:
            emotion.reset_person()

            debug_print(
                (
                    f"Active person: "
                    f"{name} "
                    f"({person_id}) "
                    f"via {source}"
                )
            )

        return changed

    def active_person():
        with state_lock:
            return (
                state[
                    "current_person_id"
                ],
                state[
                    "current_name"
                ],
            )

    def clear_active_person():
        if not identity.available:
            return

        with state_lock:
            state[
                "current_person_id"
            ] = None

            state[
                "current_name"
            ] = None

            state[
                "identity_status"
            ] = "UNKNOWN"

    def shutdown(
        *_,
    ):
        if not stop.is_set():
            print(
                (
                    f"\nStopping "
                    f"{assistant_name}..."
                ),
                flush=True,
            )

            stop.set()

            face.stop()

            vision.stop()

    signal.signal(
        signal.SIGINT,
        shutdown,
    )

    signal.signal(
        signal.SIGTERM,
        shutdown,
    )

    if not args.no_face:
        thread = threading.Thread(
            target=face.run,
            name="face",
            daemon=False,
        )

        thread.start()

        threads.append(
            thread
        )

    if not args.no_vision:
        thread = threading.Thread(
            target=vision.run,
            args=(
                face,
            ),
            name="vision",
            daemon=False,
        )

        thread.start()

        threads.append(
            thread
        )

    def keyboard():
        while not stop.is_set():
            line = None

            try:
                line = input(
                    ">"
                    if not args.debug
                    else "> "
                )

            except EOFError:
                return

            except KeyboardInterrupt:
                shutdown()
                return

            if (
                line
                and line.strip()
            ):
                messages.put(
                    (
                        "user",
                        line.strip(),
                    )
                )

    def microphone():
        while not stop.is_set():
            blocked = (
                busy.is_set()
                or speech.is_speaking.is_set()
            )

            if blocked:
                time.sleep(
                    0.05
                )

            else:
                text = speech.listen_once(
                    face.set_state,
                    stop_event=stop,
                )

                if (
                    text
                    and not stop.is_set()
                ):
                    messages.put(
                        (
                            "user",
                            text,
                        )
                    )

    def process_identity(
        scene,
    ):
        nonlocal last_identity_check

        if (
            args.no_identity
            or args.no_vision
            or not identity.available
        ):
            return

        now = time.monotonic()

        person_visible = bool(
            scene.get(
                "person_present"
            )
        )

        if not person_visible:
            identity.clear_current_identity()

            clear_active_person()

            with state_lock:
                state[
                    "unknown_since"
                ] = None

            return

        if (
            now
            - last_identity_check
            < 0.6
        ):
            return

        last_identity_check = now

        result = (
            identity
            .recognize_from_vision(
                vision
            )
        )

        if (
            result.status
            == "KNOWN"
        ):
            set_active_person(
                result.person_id,
                result.name,
                source="FACE",
            )

            wall_now = time.time()

            previous = (
                last_reinforcement.get(
                    result.person_id,
                    0.0,
                )
            )

            if (
                result.similarity
                >= identity
                .strong_match_threshold
                and wall_now
                - previous
                >= 3600.0
            ):
                reinforced = (
                    identity
                    .reinforce_identity(
                        result.person_id,
                        vision,
                    )
                )

                if reinforced:
                    last_reinforcement[
                        result.person_id
                    ] = wall_now

                    debug_print(
                        (
                            "Added identity "
                            "reinforcement for "
                            f"{result.name}"
                        )
                    )

        elif result.status in {
            "UNKNOWN",
            "LOW_QUALITY",
            "NO_FACE",
        }:
            with state_lock:
                if (
                    state[
                        "unknown_since"
                    ]
                    is None
                ):
                    state[
                        "unknown_since"
                    ] = time.time()

                state[
                    "identity_status"
                ] = result.status

        debug_print(
            "Identity:",
            result.status,
            result.name,
            (
                f"similarity="
                f"{result.similarity:.3f}"
            ),
        )

    def process_emotion(
        scene,
    ):
        if (
            args.no_emotion
            or args.no_vision
            or not emotion.available
        ):
            return

        person_id, _ = (
            active_person()
        )

        if (
            not person_id
            or not scene.get(
                "person_present"
            )
        ):
            return

        result = (
            emotion
            .observe_from_vision(
                vision,
                person_id=person_id,
            )
        )

        if (
            result.status
            == "STABLE"
        ):
            with state_lock:
                state[
                    "last_emotion"
                ] = result.as_dict()

            debug_print(
                "Face emotion:",
                result.emotion,
                (
                    f"confidence="
                    f"{result.confidence:.2f}"
                ),
            )

    def process_objects(
        scene,
    ):
        now = time.time()

        for item in scene.get(
            "objects",
            [],
        ):
            label = str(
                item
            ).lower()

            last_write = (
                last_object_write.get(
                    label,
                    0.0,
                )
            )

            if (
                now
                - last_write
                >= 30.0
            ):
                memory.see_object(
                    label
                )

                last_object_write[
                    label
                ] = now

    def awareness():
        while not stop.is_set():
            if not args.no_vision:
                scene = (
                    vision.get_scene()
                )

                process_identity(
                    scene
                )

                process_emotion(
                    scene
                )

                process_objects(
                    scene
                )

                event = behavior.update(
                    scene
                )

                if (
                    event is not None
                    and not busy.is_set()
                    and not speech
                    .is_speaking
                    .is_set()
                ):
                    messages.put(
                        (
                            "proactive",
                            event,
                        )
                    )

            time.sleep(
                0.25
            )

    keyboard_thread = (
        threading.Thread(
            target=keyboard,
            name="keyboard",
            daemon=True,
        )
    )

    keyboard_thread.start()

    if not args.no_mic:
        thread = (
            threading.Thread(
                target=microphone,
                name="microphone",
                daemon=True,
            )
        )

        thread.start()

    awareness_thread = (
        threading.Thread(
            target=awareness,
            name="awareness",
            daemon=True,
        )
    )

    awareness_thread.start()

    try:
        while not stop.is_set():
            source = None

            payload = None

            try:
                source, payload = (
                    messages.get(
                        timeout=0.2
                    )
                )

            except queue.Empty:
                pass

            if (
                source
                == "proactive"
            ):
                if not busy.is_set():
                    busy.set()

                    try:
                        speak_reply(
                            speech,
                            face,
                            payload[
                                "text"
                            ],
                            payload.get(
                                "emotion",
                                "neutral",
                            ),
                        )

                    finally:
                        busy.clear()

            elif source == "user":
                busy.set()

                try:
                    text = str(
                        payload
                    ).strip()

                    if args.debug:
                        print(
                            (
                                f"User: "
                                f"{text}"
                            ),
                            flush=True,
                        )

                    if looks_visual(
                        text
                    ):
                        vision.request_visual_attention(
                            3.0
                        )

                    (
                        person_id,
                        name,
                    ) = active_person()

                    with state_lock:
                        awaiting_name = (
                            state[
                                "awaiting_name"
                            ]
                        )

                    if awaiting_name:
                        detected_name = (
                            extract_name(
                                text
                            )
                        )

                        if detected_name:
                            existing = (
                                memory
                                .find_person_by_name(
                                    detected_name
                                )
                            )

                            if existing:
                                person_id = (
                                    existing[
                                        "id"
                                    ]
                                )

                            else:
                                person_id = (
                                    memory
                                    .create_person(
                                        detected_name,
                                        make_current=True,
                                    )
                                )

                            set_active_person(
                                person_id,
                                detected_name,
                                source=(
                                    "ONBOARDING"
                                ),
                            )

                            memory.remember(
                                (
                                    "The person's "
                                    "name is "
                                    f"{detected_name}."
                                ),
                                kind="identity",
                                importance=1.0,
                                source=(
                                    "onboarding"
                                ),
                                person_id=(
                                    person_id
                                ),
                            )

                            enrolled = False

                            sample_count = 0

                            if (
                                identity.available
                                and not args.no_identity
                            ):
                                enrollment = (
                                    identity
                                    .enroll_person_from_vision(
                                        detected_name,
                                        vision,
                                        samples=5,
                                        timeout=6.0,
                                    )
                                )

                                enrolled = bool(
                                    enrollment.get(
                                        "ok"
                                    )
                                )

                                sample_count = int(
                                    enrollment.get(
                                        "samples",
                                        0,
                                    )
                                )

                            with state_lock:
                                state[
                                    "awaiting_name"
                                ] = False

                            if enrolled:
                                onboarding_text = (
                                    f"Nice to meet you, "
                                    f"{detected_name}. "
                                    "I'll recognize you "
                                    "next time."
                                )

                            elif (
                                identity.available
                            ):
                                onboarding_text = (
                                    f"Nice to meet you, "
                                    f"{detected_name}. "
                                    "I saved your profile, "
                                    "but I need a clearer "
                                    "look at your face before "
                                    "I can recognize you "
                                    "reliably."
                                )

                            else:
                                onboarding_text = (
                                    f"Nice to meet you, "
                                    f"{detected_name}. "
                                    "I'll remember your "
                                    "profile. Face recognition "
                                    "will activate when its "
                                    "model is installed."
                                )

                            debug_print(
                                (
                                    "Enrollment samples "
                                    f"for {detected_name}: "
                                    f"{sample_count}"
                                )
                            )

                            speak_reply(
                                speech,
                                face,
                                onboarding_text,
                                "happy",
                            )

                        else:
                            speak_reply(
                                speech,
                                face,
                                (
                                    "I didn't catch your "
                                    "name. Just say, "
                                    "'My name is Vlad.'"
                                ),
                                "confused",
                            )

                    elif not name:
                        with state_lock:
                            state[
                                "awaiting_name"
                            ] = True

                        speak_reply(
                            speech,
                            face,
                            (
                                "Hey. I don't think "
                                "we've met yet. "
                                "What's your name?"
                            ),
                            "curious",
                        )

                    elif asks_forget_person(
                        text
                    ):
                        old_name = name

                        if identity.available:
                            identity.delete_face_data(
                                person_id
                            )

                        memory.forget_person(
                            person_id
                        )

                        emotion.reset_person(
                            person_id
                        )

                        with state_lock:
                            state[
                                "current_person_id"
                            ] = None

                            state[
                                "current_name"
                            ] = None

                            state[
                                "identity_status"
                            ] = "UNKNOWN"

                            state[
                                "awaiting_name"
                            ] = False

                        speak_reply(
                            speech,
                            face,
                            (
                                "Okay. I deleted "
                                "the profile and "
                                "memories I had for "
                                f"{old_name}."
                            ),
                            "neutral",
                        )

                    elif asks_memory_summary(
                        text
                    ):
                        summary = (
                            memory
                            .describe_person(
                                person_id=(
                                    person_id
                                )
                            )
                        )

                        emotion_context = (
                            emotion
                            .get_context(
                                person_id=(
                                    person_id
                                )
                            )
                        )

                        context = summary

                        if emotion_context:
                            context += (
                                "\nTemporary "
                                "emotional context: "
                                + str(
                                    emotion_context
                                )
                            )

                        reply = ai.ask(
                            (
                                "Tell me naturally "
                                "what you remember "
                                "about me. Do not "
                                "invent anything."
                            ),
                            scene=(
                                vision.get_scene()
                            ),
                            person_name=name,
                            memory_context=(
                                context
                            ),
                        )

                        speak_reply(
                            speech,
                            face,
                            reply[
                                "text"
                            ],
                            reply.get(
                                "emotion",
                                "neutral",
                            ),
                        )

                    elif asks_forget_last(
                        text
                    ):
                        forgotten = (
                            memory
                            .forget_last(
                                person_id=(
                                    person_id
                                )
                            )
                        )

                        if forgotten:
                            response = (
                                "Okay. I won't use "
                                "that memory anymore."
                            )

                        else:
                            response = (
                                "I don't have "
                                "anything recent "
                                "to forget."
                            )

                        speak_reply(
                            speech,
                            face,
                            response,
                            "neutral",
                        )

                    else:
                        explicit_emotion = (
                            detect_explicit_emotion(
                                text
                            )
                        )

                        if (
                            explicit_emotion
                            and not args.no_emotion
                        ):
                            if person_id:
                                emotion.set_explicit_emotion(
                                    person_id=person_id,
                                    emotion=explicit_emotion,
                                    text=text,
                                    confidence=0.95,
                                )

                            with state_lock:
                                state[
                                    "last_emotion"
                                ] = {
                                    "emotion": (
                                        explicit_emotion
                                    ),
                                    "source": (
                                        "explicit_speech"
                                    ),
                                    "confidence": 0.95,
                                    "updated_at": time.time(),
                                }

                            debug_print(
                                (
                                    "Explicit emotion: "
                                    f"{explicit_emotion}"
                                )
                            )

                        memory_context = (
                            memory.context(
                                text,
                                person_id=(
                                    person_id
                                ),
                            )
                        )

                        emotion_context = (
                            emotion.get_context(
                                person_id=(
                                    person_id
                                )
                            )
                        )

                        if not emotion_context:
                            with state_lock:
                                recent_emotion = state.get(
                                    "last_emotion"
                                )

                            if recent_emotion:
                                updated_at = float(
                                    recent_emotion.get(
                                        "updated_at",
                                        0.0,
                                    )
                                )

                                if (
                                    updated_at > 0.0
                                    and time.time() - updated_at <= 600.0
                                ):
                                    emotion_context = dict(
                                        recent_emotion
                                    )

                        if emotion_context:
                            memory_context += (
                                "\nTemporary "
                                "emotional context: "
                                + str(
                                    emotion_context
                                )
                            )

                        if (
                            looks_visual(text)
                            and not args.no_vision
                        ):
                            image_bytes = vision.snapshot_jpeg(
                                quality=85
                            )

                            reply = ai.ask_visual(
                                text,
                                image_bytes=image_bytes,
                                person_name=name,
                                memory_context=memory_context,
                                emotional_context=emotion_context,
                            )
                        else:
                            reply = ai.ask(
                                text,
                                scene=vision.get_scene(),
                                person_name=name,
                                memory_context=memory_context,
                                emotional_context=emotion_context,
                            )

                        memory_request = (
                            reply.get(
                                "memory",
                                {},
                            )
                        )

                        if (
                            memory_request.get(
                                "save"
                            )
                            and memory_request.get(
                                "text"
                            )
                        ):
                            memory_id = (
                                memory.remember(
                                    memory_request[
                                        "text"
                                    ],
                                    kind=(
                                        memory_request
                                        .get(
                                            "kind",
                                            "fact",
                                        )
                                    ),
                                    emotion=(
                                        memory_request
                                        .get(
                                            "emotion",
                                            "neutral",
                                        )
                                    ),
                                    importance=(
                                        memory_request
                                        .get(
                                            "importance",
                                            0.5,
                                        )
                                    ),
                                    source=(
                                        "conversation"
                                    ),
                                    person_id=(
                                        person_id
                                    ),
                                    confidence=(
                                        memory_request
                                        .get(
                                            "confidence",
                                            1.0,
                                        )
                                    ),
                                )
                            )

                            if (
                                args.debug
                                and memory_id
                            ):
                                print(
                                    (
                                        "Memory saved: "
                                        + memory_request[
                                            "text"
                                        ]
                                    ),
                                    flush=True,
                                )

                        speak_reply(
                            speech,
                            face,
                            reply[
                                "text"
                            ],
                            reply.get(
                                "emotion",
                                "neutral",
                            ),
                        )

                except Exception as exc:
                    print(
                        (
                            "Conversation "
                            f"error: {exc}"
                        ),
                        flush=True,
                    )

                    face.set_state(
                        "concerned"
                    )

                finally:
                    busy.clear()

    finally:
        shutdown()

        ai.close()

        deadline = (
            time.monotonic()
            + 4.0
        )

        for thread in threads:
            remaining = (
                deadline
                - time.monotonic()
            )

            if remaining > 0:
                thread.join(
                    timeout=remaining
                )

        print(
            (
                f"{assistant_name} "
                "stopped."
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()