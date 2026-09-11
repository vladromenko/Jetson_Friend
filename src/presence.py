"""Temporal presence from fresh observations. No inference model or persistence writes."""

from dataclasses import dataclass


@dataclass
class PersonState:
    person_id: str
    status: str = "ABSENT"
    first_seen: float = 0.0
    last_seen: float = 0.0
    candidate_since: float | None = None
    absent_since: float | None = None
    track_id: int | None = None
    confidence: float = 0.0
    confirmed: bool = False


class Presence:
    def __init__(self, confirm=1.0, absence=5.0):
        self.confirm = confirm
        self.absence = absence
        self.people = {}

    def update(self, observations, now):
        events = []
        visible = {p["person_id"]: p for p in observations if p.get("person_id")}
        for pid, observation in visible.items():
            state = self.people.setdefault(pid, PersonState(pid, first_seen=now))
            state.track_id = observation.get("track_id")
            state.confidence = observation.get("confidence", 0.0)
            state.last_seen = now
            if state.status in {"VISIBLE", "TEMPORARILY_LOST"}:
                state.status = "VISIBLE"
                continue
            if state.candidate_since is None:
                state.candidate_since = now
            if now - state.candidate_since >= self.confirm:
                kind = "person_returned" if state.confirmed else "known_person_seen"
                absence = (
                    now - state.absent_since if state.absent_since is not None else 0.0
                )
                events.append(
                    {
                        "kind": kind,
                        "person_id": pid,
                        "absence": absence,
                        "created": now,
                        "confidence": state.confidence,
                    }
                )
                state.status = "VISIBLE"
                state.confirmed = True
                state.candidate_since = None
                state.absent_since = None
        for pid, state in self.people.items():
            if pid in visible:
                continue
            state.candidate_since = None
            if state.status == "VISIBLE":
                state.status = "TEMPORARILY_LOST"
            if (
                state.status == "TEMPORARILY_LOST"
                and now - state.last_seen >= self.absence
            ):
                state.status = "ABSENT"
                state.absent_since = state.last_seen
                events.append({"kind": "person_left", "person_id": pid, "created": now})
        # Working state is bounded; persistent last_seen belongs in SQLite.
        for pid in list(self.people):
            if now - self.people[pid].last_seen > 7 * 86400:
                del self.people[pid]
        return events
