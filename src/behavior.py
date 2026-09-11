"""Conservative policy: perception updates state; only meaningful returns may speak."""

import os
import threading
import time

from presence import Presence


class Behavior:
    def __init__(self, memory):
        self.memory = memory
        self.lock = threading.RLock()
        self.presence = Presence(
            float(os.getenv("BEHAVIOR_PRESENCE_CONFIRM_SEC", "1")),
            float(os.getenv("BEHAVIOR_ABSENCE_CONFIRM_SEC", "5")),
        )
        self.long_absence = float(os.getenv("LONG_ABSENCE_HOURS", "6")) * 3600
        self.return_cooldown = float(os.getenv("RETURN_COOLDOWN_HOURS", "4")) * 3600
        self.quiet_seconds = float(os.getenv("PROACTIVE_QUIET_SEC", "60"))
        self.last_spoke = float("-inf")
        self.last_return_message = {}
        self.pending = None
        self.last_events = []

    def note_speech(self):
        with self.lock:
            self.last_spoke = time.monotonic()

    def update(self, scene, busy=False, listening=False, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            fresh = time.time() - scene.get("updated", 0) <= 2.0
            people = scene.get("people", []) if fresh else []
            self.last_events = self.presence.update(people, now)
            for event in self.last_events:
                if (
                    event["kind"] == "person_returned"
                    and event["absence"] >= self.long_absence
                ):
                    self.pending = event
            event = self.pending
            if event is None:
                return None
            pid = event["person_id"]
            # A greeting is obsolete if its person left, or multiple people are nearby.
            if (
                now - event["created"] > 15
                or len(people) != 1
                or people[0].get("person_id") != pid
            ):
                self.pending = None
                return None
            if busy or listening or now - self.last_spoke < self.quiet_seconds:
                return None
            last = self.last_return_message.get(pid, float("-inf"))
            if now - last < self.return_cooldown:
                self.pending = None
                return None
            person = self.memory.get_person(pid)
            if not person:
                self.pending = None
                return None
            # No LLM needed for a short return acknowledgement.
            return {
                **event,
                "text": f"Hey {person['name']}, you're back.",
                "emotion": "happy",
            }

    def acknowledge(self, event):
        """Commit cooldown only when the caller actually attempts the action."""
        with self.lock:
            self.last_return_message[event["person_id"]] = time.monotonic()
            self.pending = None
            self.note_speech()

    def forget_person(self, person_id):
        with self.lock:
            self.presence.people.pop(person_id, None)
            self.last_return_message.pop(person_id, None)
            if self.pending and self.pending["person_id"] == person_id:
                self.pending = None

    def get_state(self):
        with self.lock:
            return {
                "people": {
                    pid: vars(state).copy()
                    for pid, state in self.presence.people.items()
                },
                "events": list(self.last_events),
            }
