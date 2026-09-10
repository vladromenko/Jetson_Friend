import os
import threading
import time


class Behavior:
    """
    Lightweight initiative manager for MILO.

    It turns noisy detector observations into slow, conservative activity
    states and emits only occasional proactive events. It intentionally does
    not make strong psychological conclusions from vision.
    """

    def __init__(self, memory):
        self.memory = memory
        self.lock = threading.RLock()

        now = time.time()

        self.person_present = False
        self.last_person_seen = 0.0
        self.absence_started = now
        self.presence_started = 0.0

        self.activity = "away"
        self.activity_started = now

        self.last_phone_seen = 0.0
        self.last_computer_seen = 0.0

        self.last_return_message = {}
        self.last_phone_message = {}
        self.last_computer_message = {}

        self.phone_limit = float(
            os.getenv(
                "PHONE_REMINDER_MINUTES",
                "20",
            )
        ) * 60.0

        self.computer_limit = float(
            os.getenv(
                "COMPUTER_BREAK_MINUTES",
                "60",
            )
        ) * 60.0

        self.long_absence = float(
            os.getenv(
                "LONG_ABSENCE_HOURS",
                "6",
            )
        ) * 3600.0

        self.return_cooldown = float(
            os.getenv(
                "RETURN_COOLDOWN_HOURS",
                "4",
            )
        ) * 3600.0

        self.reminder_cooldown = float(
            os.getenv(
                "ACTIVITY_REMINDER_COOLDOWN_MINUTES",
                "45",
            )
        ) * 60.0

        self.object_hold = float(
            os.getenv(
                "BEHAVIOR_OBJECT_HOLD_SEC",
                "8",
            )
        )

        self.presence_confirm = float(
            os.getenv(
                "BEHAVIOR_PRESENCE_CONFIRM_SEC",
                "1.0",
            )
        )

        self.absence_confirm = float(
            os.getenv(
                "BEHAVIOR_ABSENCE_CONFIRM_SEC",
                "4.0",
            )
        )

        self.activity_min_log = float(
            os.getenv(
                "BEHAVIOR_ACTIVITY_LOG_MINUTES",
                "5",
            )
        ) * 60.0

        self._visible_candidate_since = None
        self._missing_candidate_since = None

    def _person_id(self):
        try:
            return self.memory.get_current_person_id()
        except Exception:
            return None

    @staticmethod
    def _person_key(person_id):
        return person_id or "__unknown__"

    @staticmethod
    def _object_names(scene):
        names = set()

        for item in scene.get("objects", []):
            if isinstance(item, dict):
                label = (
                    item.get("label")
                    or item.get("name")
                )
            else:
                label = item

            if label:
                names.add(
                    str(label)
                    .strip()
                    .lower()
                )

        return names

    @staticmethod
    def _visible_now(scene, objects):
        if bool(
            scene.get(
                "person_present"
            )
        ):
            return True

        faces = scene.get(
            "faces",
            0,
        )

        if isinstance(
            faces,
            list,
        ):
            if len(faces) > 0:
                return True

        else:
            try:
                if int(faces) > 0:
                    return True

            except (
                TypeError,
                ValueError,
            ):
                pass

        return (
            "person"
            in objects
        )

    def _confirmed_presence(
        self,
        visible_now,
        now,
    ):
        if visible_now:
            self._missing_candidate_since = None

            if self.person_present:
                self.last_person_seen = now
                return True

            if (
                self._visible_candidate_since
                is None
            ):
                self._visible_candidate_since = now

            if (
                now
                - self._visible_candidate_since
                >= self.presence_confirm
            ):
                self.person_present = True
                self.last_person_seen = now
                self.presence_started = now
                self._visible_candidate_since = None
                return True

            return False

        self._visible_candidate_since = None

        if not self.person_present:
            return False

        if (
            self._missing_candidate_since
            is None
        ):
            self._missing_candidate_since = now

        if (
            now
            - self._missing_candidate_since
            >= self.absence_confirm
        ):
            self.person_present = False
            self.absence_started = now
            self._missing_candidate_since = None

            self._set_activity(
                "away",
                now,
                self._person_id(),
            )

            return False

        return True

    def update(self, scene):
        now = time.time()

        with self.lock:
            objects = (
                self._object_names(
                    scene
                )
            )

            visible_raw = (
                self._visible_now(
                    scene,
                    objects,
                )
            )

            was_present = (
                self.person_present
            )

            visible = (
                self._confirmed_presence(
                    visible_raw,
                    now,
                )
            )

            person_id = (
                self._person_id()
            )

            person_key = (
                self._person_key(
                    person_id
                )
            )

            event = None

            if (
                visible
                and not was_present
            ):
                absence = max(
                    0.0,
                    now
                    - self.absence_started,
                )

                last_return = (
                    self.last_return_message
                    .get(
                        person_key,
                        0.0,
                    )
                )

                if (
                    person_id
                    and absence
                    >= self.long_absence
                    and now
                    - last_return
                    >= self.return_cooldown
                ):
                    hours = (
                        absence
                        / 3600.0
                    )

                    event = {
                        "kind": "return",
                        "text": (
                            self._return_text(
                                hours
                            )
                        ),
                        "emotion": "happy",
                        "person_id": (
                            person_id
                        ),
                    }

                    self.last_return_message[
                        person_key
                    ] = now

                    self.memory.event(
                        "return",
                        (
                            "Person returned "
                            "after about "
                            f"{hours:.1f} "
                            "hours away."
                        ),
                        0.4,
                        person_id=(
                            person_id
                        ),
                    )

            self._update_object_activity_timestamps(
                objects,
                now,
            )

            new_activity = (
                self._classify(
                    visible,
                    now,
                )
            )

            if (
                new_activity
                != self.activity
            ):
                self._set_activity(
                    new_activity,
                    now,
                    person_id,
                )

            if event is None:
                event = (
                    self._activity_event(
                        now,
                        person_id,
                        person_key,
                    )
                )

            return event

    @staticmethod
    def _return_text(hours):
        if hours >= 24.0:
            return (
                "Hey, you're back. "
                "I haven't seen you "
                "in quite a while. "
                "You alive? Everything okay?"
            )

        return (
            "Hey, you're back. "
            "I haven't seen you for a while. "
            "Everything okay?"
        )

    def _update_object_activity_timestamps(
        self,
        objects,
        now,
    ):
        if (
            "cell phone"
            in objects
            or "phone"
            in objects
            or "mobile phone"
            in objects
        ):
            self.last_phone_seen = now

        if (
            "laptop"
            in objects
            or "keyboard"
            in objects
            or "mouse"
            in objects
            or "computer"
            in objects
            or "monitor"
            in objects
        ):
            self.last_computer_seen = now

    def _classify(
        self,
        visible,
        now,
    ):
        if not visible:
            return "away"

        if (
            self.last_phone_seen
            > 0.0
            and now
            - self.last_phone_seen
            <= self.object_hold
        ):
            return "phone"

        if (
            self.last_computer_seen
            > 0.0
            and now
            - self.last_computer_seen
            <= self.object_hold
        ):
            return "computer"

        return "present"

    def _activity_event(
        self,
        now,
        person_id,
        person_key,
    ):
        if not person_id:
            return None

        duration = max(
            0.0,
            now
            - self.activity_started,
        )

        if self.activity == "phone":
            last_message = (
                self.last_phone_message
                .get(
                    person_key,
                    0.0,
                )
            )

            if (
                duration
                >= self.phone_limit
                and now
                - last_message
                >= self.reminder_cooldown
            ):
                minutes = max(
                    1,
                    int(
                        duration
                        / 60.0
                    ),
                )

                self.last_phone_message[
                    person_key
                ] = now

                self.memory.event(
                    "activity_reminder",
                    (
                        "Phone-use reminder "
                        "after about "
                        f"{minutes} minutes."
                    ),
                    0.2,
                    person_id=(
                        person_id
                    ),
                )

                return {
                    "kind": "phone",
                    "text": (
                        "You've been on your "
                        "phone for a while. "
                        "If you meant to be "
                        "working, want to get "
                        "back to it?"
                    ),
                    "emotion": "curious",
                    "person_id": (
                        person_id
                    ),
                }

        if (
            self.activity
            == "computer"
        ):
            last_message = (
                self.last_computer_message
                .get(
                    person_key,
                    0.0,
                )
            )

            if (
                duration
                >= self.computer_limit
                and now
                - last_message
                >= self.reminder_cooldown
            ):
                minutes = max(
                    1,
                    int(
                        duration
                        / 60.0
                    ),
                )

                self.last_computer_message[
                    person_key
                ] = now

                self.memory.event(
                    "activity_reminder",
                    (
                        "Computer-break "
                        "reminder after about "
                        f"{minutes} minutes."
                    ),
                    0.2,
                    person_id=(
                        person_id
                    ),
                )

                return {
                    "kind": (
                        "computer_break"
                    ),
                    "text": (
                        "You've been at the "
                        "computer for quite "
                        "a while. A short "
                        "stretch or walk might "
                        "be a good idea."
                    ),
                    "emotion": "curious",
                    "person_id": (
                        person_id
                    ),
                }

        return None

    def _set_activity(
        self,
        activity,
        now,
        person_id=None,
    ):
        previous = (
            self.activity
        )

        duration = max(
            0.0,
            now
            - self.activity_started,
        )

        if (
            previous
            not in {
                "away",
                "none",
                "present",
            }
            and duration
            >= self.activity_min_log
        ):
            self.memory.event(
                "activity",
                (
                    f"Observed activity "
                    f"'{previous}' for "
                    "about "
                    f"{int(duration / 60.0)} "
                    "minutes."
                ),
                0.15,
                person_id=(
                    person_id
                ),
            )

        self.activity = (
            activity
        )

        self.activity_started = (
            now
        )

    def get_state(self):
        with self.lock:
            return {
                "person_present": (
                    self.person_present
                ),
                "activity": (
                    self.activity
                ),
                "activity_seconds": max(
                    0.0,
                    time.time()
                    - self.activity_started,
                ),
                "last_person_seen": (
                    self.last_person_seen
                ),
                "absence_started": (
                    self.absence_started
                ),
            }