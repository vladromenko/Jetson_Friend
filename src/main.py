import argparse
import os
import queue
import re
import signal
import threading
import time

from ai import HughAI
from behavior import Behavior
from face import Face
from memory import Memory
from speech import Speech
from vision import Vision


def speak_reply(
    speech,
    face,
    text,
    emotion="neutral",
):
    face.set_state(
        emotion
    )

    time.sleep(
        0.15
    )

    speech.speak(
        text,
        face.set_state,
    )

    face.set_state(
        emotion
    )

    time.sleep(
        0.5
    )

    face.set_state(
        "neutral"
    )


def extract_name(text):
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\-']{1,30})",
        r"\bi am\s+([A-Za-z][A-Za-z\-']{1,30})",
        r"\bi'm\s+([A-Za-z][A-Za-z\-']{1,30})",
        r"\bcall me\s+([A-Za-z][A-Za-z\-']{1,30})",
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


def save_reference_face(
    vision,
    name,
):
    root = os.getenv(
        "JETSON_FRIEND_ROOT",
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
    )

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        name.lower(),
    )

    directory = os.path.join(
        root,
        "data",
        "people",
        safe_name,
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    image = vision.snapshot_jpeg(
        quality=92
    )

    if image is None:
        return None

    path = os.path.join(
        directory,
        "reference.jpg",
    )

    with open(
        path,
        "wb",
    ) as file:
        file.write(
            image
        )

    return path


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

    args = parser.parse_args()

    memory = Memory()
    behavior = Behavior(
        memory
    )
    face = Face()
    speech = Speech()
    vision = Vision()

    print(
        "Starting Hugh...",
        flush=True,
    )

    ai = HughAI()

    stop = threading.Event()
    busy = threading.Event()

    messages = queue.Queue()

    threads = []

    state = {
        "awaiting_name": False,
    }

    last_object_write = {}

    def shutdown(*_):
        if stop.is_set():
            return

        print(
            "\nStopping Hugh...",
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
            args=(face,),
            name="vision",
            daemon=False,
        )

        thread.start()
        threads.append(
            thread
        )

    def keyboard():
        while not stop.is_set():
            try:
                line = input(
                    "> "
                    if args.debug
                    else ""
                )
            except EOFError:
                return
            except KeyboardInterrupt:
                shutdown()
                return

            if line.strip():
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

    def awareness():
        while not stop.is_set():
            if not args.no_vision:
                scene = dict(
                    vision.scene
                )

                now = time.time()

                for item in scene.get(
                    "objects",
                    [],
                ):
                    label = str(
                        item
                    ).lower()

                    last_write = last_object_write.get(
                        label,
                        0.0,
                    )

                    if (
                        now - last_write
                        >= 30.0
                    ):
                        memory.see_object(
                            label
                        )

                        last_object_write[
                            label
                        ] = now

                event = behavior.update(
                    scene
                )

                if (
                    event is not None
                    and not busy.is_set()
                    and not speech.is_speaking.is_set()
                ):
                    messages.put(
                        (
                            "proactive",
                            event,
                        )
                    )

            time.sleep(
                1.0
            )

    keyboard_thread = threading.Thread(
        target=keyboard,
        name="keyboard",
        daemon=True,
    )

    keyboard_thread.start()

    if not args.no_mic:
        thread = threading.Thread(
            target=microphone,
            name="microphone",
            daemon=True,
        )

        thread.start()

    awareness_thread = threading.Thread(
        target=awareness,
        name="awareness",
        daemon=True,
    )

    awareness_thread.start()

    try:
        while not stop.is_set():
            try:
                source, payload = messages.get(
                    timeout=0.2
                )
            except queue.Empty:
                source = None
                payload = None

            if source == "proactive":
                if not busy.is_set():
                    busy.set()

                    try:
                        speak_reply(
                            speech,
                            face,
                            payload["text"],
                            payload.get(
                                "emotion",
                                "neutral",
                            ),
                        )
                    finally:
                        busy.clear()

            if source == "user":
                busy.set()

                try:
                    text = str(
                        payload
                    ).strip()

                    if args.debug:
                        print(
                            f"User: {text}",
                            flush=True,
                        )

                    name = memory.get_profile(
                        "name"
                    )

                    if state[
                        "awaiting_name"
                    ]:
                        detected_name = extract_name(
                            text
                        )

                        if detected_name:
                            memory.set_profile(
                                "name",
                                detected_name,
                            )

                            memory.remember(
                                (
                                    f"The person's name is "
                                    f"{detected_name}."
                                ),
                                kind="identity",
                                importance=1.0,
                                source="onboarding",
                            )

                            photo = save_reference_face(
                                vision,
                                detected_name,
                            )

                            if photo:
                                memory.set_profile(
                                    "reference_face",
                                    photo,
                                )

                            state[
                                "awaiting_name"
                            ] = False

                            speak_reply(
                                speech,
                                face,
                                (
                                    f"Nice to meet you, {detected_name}. "
                                    "I'll remember you."
                                ),
                                "happy",
                            )

                        else:
                            speak_reply(
                                speech,
                                face,
                                (
                                    "I didn't catch your name. "
                                    "Just say something like, "
                                    "'My name is Vlad.'"
                                ),
                                "confused",
                            )

                    elif not name:
                        state[
                            "awaiting_name"
                        ] = True

                        speak_reply(
                            speech,
                            face,
                            (
                                "Hey. I don't think we've properly met yet. "
                                "What's your name?"
                            ),
                            "curious",
                        )

                    elif re.search(
                        r"\bwhat do you remember about me\b",
                        text,
                        flags=re.I,
                    ):
                        summary = memory.describe_person()

                        reply = ai.ask(
                            (
                                "Tell me naturally what you remember about me. "
                                "Do not invent anything."
                            ),
                            scene=vision.scene,
                            person_name=name,
                            memory_context=summary,
                        )

                        speak_reply(
                            speech,
                            face,
                            reply["text"],
                            reply["emotion"],
                        )

                    elif re.search(
                        r"\bforget that\b|\bdon't remember that\b",
                        text,
                        flags=re.I,
                    ):
                        forgotten = memory.forget_last()

                        if forgotten:
                            response = (
                                "Okay. I won't use that memory anymore."
                            )
                        else:
                            response = (
                                "I don't have anything recent to forget."
                            )

                        speak_reply(
                            speech,
                            face,
                            response,
                            "neutral",
                        )

                    else:
                        memory_context = memory.context(
                            text
                        )

                        reply = ai.ask(
                            text,
                            scene=vision.scene,
                            person_name=name,
                            memory_context=memory_context,
                        )

                        memory_request = reply.get(
                            "memory",
                            {},
                        )

                        if (
                            memory_request.get(
                                "save"
                            )
                            and memory_request.get(
                                "text"
                            )
                        ):
                            memory.remember(
                                memory_request[
                                    "text"
                                ],
                                kind=memory_request.get(
                                    "kind",
                                    "fact",
                                ),
                                emotion=memory_request.get(
                                    "emotion",
                                    "neutral",
                                ),
                                importance=memory_request.get(
                                    "importance",
                                    0.5,
                                ),
                                source="conversation",
                            )

                            if args.debug:
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
                            reply["text"],
                            reply["emotion"],
                        )

                except Exception as exc:
                    print(
                        f"Conversation error: {exc}",
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
            "Hugh stopped.",
            flush=True,
        )


if __name__ == "__main__":
    main()
