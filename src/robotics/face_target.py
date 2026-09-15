"""Smoothed, detector-only target. No inference or hardware publishing here."""
import math
from dataclasses import dataclass
from .face_search import FaceSearch


@dataclass(frozen=True)
class FaceTarget:
    box: tuple | None
    center: tuple | None
    confidence: float | None
    track_id: int
    timestamp: float
    state: str
    error: tuple
    correction_deg: float


class FaceTargetTracker:
    def __init__(self, target_x=0.5, deadzone=0.06, sign=1, alpha=0.25,
                 min_confidence=0.55, interval=0.5):
        values = (target_x, deadzone, alpha, min_confidence, interval)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite tracker configuration')
        if not (0 <= target_x <= 1 and 0.03 <= deadzone <= 0.2 and
                0 < alpha <= 1 and 0 <= min_confidence <= 1 and interval >= 0.5 and sign in (-1, 1)):
            raise ValueError('Invalid tracker configuration')
        self.search = FaceSearch()
        self.target_x, self.deadzone, self.sign = target_x, deadzone, sign
        self.alpha, self.min_confidence, self.interval = alpha, min_confidence, interval
        self.center = None
        self.track_id = 0
        self.next_correction = 0.0
        self.last_stamp = -math.inf

    def update(self, detections, stamp, now):
        if not all(math.isfinite(t) for t in (stamp, now)):
            raise ValueError('Invalid timestamp')
        fresh = 0 <= now - stamp <= 0.75 and stamp > self.last_stamp
        accepted = []
        if fresh:
            self.last_stamp = stamp
            for box, confidence in detections:
                valid = len(box) == 4 and all(math.isfinite(v) and 0 <= v <= 1 for v in box)
                valid = valid and box[0] < box[2] and box[1] < box[3]
                score_ok = confidence is None or (math.isfinite(confidence) and self.min_confidence <= confidence <= 1)
                if valid and score_ok:
                    accepted.append((tuple(box), confidence))
        previous_seen = self.search.last_seen
        self.search.observe([b for b, _ in accepted], stamp if fresh else self.search.last_frame, now)
        matched = fresh and self.search.last_seen == now and bool(accepted)
        if self.search.box is None:
            self.center = None
        if matched:
            new_track = previous_seen is None or now - previous_seen >= self.search.loss_timeout or self.center is None
            raw = ((self.search.box[0] + self.search.box[2])/2, (self.search.box[1] + self.search.box[3])/2)
            if new_track:
                self.track_id += 1
                self.center = raw
            else:
                self.center = tuple(a + self.alpha*(b-a) for a, b in zip(self.center, raw))
        tracked = matched and self.search.mode == 'FACE_FOUND'
        error = (0.0, 0.0) if self.center is None else (self.center[0]-self.target_x, self.center[1]-0.5)
        correction = 0.0
        # Haar has no calibrated confidence: diagnostic tracking only.
        confidence = next((c for b, c in accepted if b == self.search.box), None) if matched else None
        if tracked and confidence is not None and now >= self.next_correction and abs(error[0]) > self.deadzone:
            correction = self.sign * (1.0 if error[0] > 0 else -1.0)
            self.next_correction = now + self.interval
        return FaceTarget(self.search.box, self.center, confidence, self.track_id, stamp,
                          'tracked' if tracked else ('acquiring' if matched else 'lost'), error, correction)
