import os
import time


class Behavior:
    def __init__(self, memory):
        self.memory = memory

        self.last_person_seen = 0.0
        self.absence_started = time.time()
        self.person_present = False

        self.activity = "none"
        self.activity_started = time.time()

        self.last_return_message = 0.0
        self.last_phone_message = 0.0
        self.last_computer_message = 0.0

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

        self.reminder_cooldown = 45.0 * 60.0
        self.return_cooldown = 4.0 * 3600.0

    def update(self, scene):
        now = time.time()

        objects = {
            str(item).lower()
            for item in scene.get(
                "objects",
                [],
            )
        }

        faces = int(
            scene.get(
                "faces",
                0,
            )
        )

        visible = (
            faces > 0
            or "person" in objects
        )

        event = None

        if visible and not self.person_present:
            absence = now - self.absence_started

            self.person_present = True
            self.last_person_seen = now

            if (
                absence >= self.long_absence
                and now - self.last_return_message >= self.return_cooldown
            ):
                hours = absence / 3600.0

                event = {
                    "kind": "return",
                    "text": (
                        "Hey, I haven't seen you for quite a while. "
                        "You alive? Everything okay?"
                    ),
                    "emotion": "happy",
                }

                self.last_return_message = now

                self.memory.event(
                    "return",
                    f"Person returned after about {hours:.1f} hours away.",
                    0.5,
                )

        if visible:
            self.last_person_seen = now

        if not visible and self.person_present:
            self.person_present = False
            self.absence_started = now
            self._set_activity("away", now)

        new_activity = self._classify(
            visible,
            objects,
        )

        if new_activity != self.activity:
            self._set_activity(
                new_activity,
                now,
            )

        duration = now - self.activity_started

        if (
            event is None
            and self.activity == "phone"
            and duration >= self.phone_limit
            and now - self.last_phone_message >= self.reminder_cooldown
        ):
            minutes = int(duration / 60)

            event = {
                "kind": "phone",
                "text": (
                    f"You've been on your phone for about {minutes} minutes. "
                    "If we're supposed to be working, put it down and get back to it."
                ),
                "emotion": "concerned",
            }

            self.last_phone_message = now

            self.memory.event(
                "activity_reminder",
                f"Phone use reminder after {minutes} minutes.",
                0.3,
            )

        if (
            event is None
            and self.activity == "computer"
            and duration >= self.computer_limit
            and now - self.last_computer_message >= self.reminder_cooldown
        ):
            minutes = int(duration / 60)

            event = {
                "kind": "computer_break",
                "text": (
                    f"You've been at the computer for about {minutes} minutes. "
                    "Stand up and walk around for a few minutes."
                ),
                "emotion": "concerned",
            }

            self.last_computer_message = now

            self.memory.event(
                "activity_reminder",
                f"Computer break reminder after {minutes} minutes.",
                0.3,
            )

        return event

    def _classify(
        self,
        visible,
        objects,
    ):
        if not visible:
            return "away"

        if (
            "cell phone" in objects
            or "phone" in objects
        ):
            return "phone"

        if (
            "laptop" in objects
            or "keyboard" in objects
            or "mouse" in objects
        ):
            return "computer"

        return "present"

    def _set_activity(
        self,
        activity,
        now,
    ):
        previous = self.activity
        duration = now - self.activity_started

        if (
            previous not in {"none", "away"}
            and duration >= 120
        ):
            self.memory.event(
                "activity",
                (
                    f"Activity '{previous}' lasted "
                    f"about {int(duration / 60)} minutes."
                ),
                0.2,
            )

        self.activity = activity
        self.activity_started = now
