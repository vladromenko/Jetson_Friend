"""Detector-only face acquisition. All times are monotonic; boxes are normalized."""
import math
from dataclasses import dataclass


@dataclass
class FaceSearch:
    minimum: float = 45
    maximum: float = 135
    step: float = 3
    runtime_ms: int = 320
    settle: float = 0.10
    direction: int = 1
    required_hits: int = 3
    loss_timeout: float = 2.5

    def __post_init__(self):
        values = (self.minimum, self.maximum, self.step, self.runtime_ms,
                  self.settle, self.loss_timeout)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Face search configuration must be finite")
        if not (0 <= self.minimum < self.maximum <= 180 and 1 <= self.step <= 3):
            raise ValueError("Face search requires J1 bounds within 0..180 and step 1..3")
        if not (320 <= self.runtime_ms <= 5000 and 0.1 <= self.settle <= 5):
            raise ValueError("Face search runtime must be 320..5000 ms; settle 0.1..5 s")
        if any(v != int(v) for v in (self.minimum, self.maximum, self.step)):
            raise ValueError("J1 bounds and step must be whole degrees")
        if self.direction not in (-1, 1) or not 3 <= self.required_hits <= 20:
            raise ValueError("Face search direction must be -1/+1 and confirmations 3..20")
        if not 1 <= self.loss_timeout <= 30:
            raise ValueError("Face loss timeout must be 1..30 seconds")
        self.reset()

    def reset(self):
        self.mode = "FACE_SEARCH"
        self.box = None
        self.hits = 0
        self.last_seen = None
        self.last_frame = None

    @staticmethod
    def area(box):
        return max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))

    def matches(self, box):
        old = self.box
        distance = math.hypot((old[0] + old[2] - box[0] - box[2]) / 2,
                              (old[1] + old[3] - box[1] - box[3]) / 2)
        return distance <= 0.18 and 0.45 <= self.area(box) / self.area(old) <= 2.2

    def observe(self, faces, stamp, now):
        # A repeated image is never another confirmation and never renews a lock.
        if self.last_seen is not None and now - self.last_seen >= self.loss_timeout:
            self.box = None
            self.hits = 0
            self.mode = "FACE_SEARCH"
            self.last_seen = None
        if stamp == self.last_frame:
            return self.mode
        self.last_frame = stamp
        candidates = faces if self.box is None else [f for f in faces if self.matches(f)]
        if not candidates:
            if self.mode != "FACE_FOUND":
                self.box = None
                self.hits = 0
            return self.mode
        self.box = max(candidates, key=self.area)
        self.last_seen = now
        self.hits += 1
        if self.hits >= self.required_hits:
            self.mode = "FACE_FOUND"
        return self.mode

    def next_target(self, commanded):
        if not self.minimum <= commanded <= self.maximum:
            raise ValueError("Commanded J1 is outside face-search sector; pose required")
        if commanded >= self.maximum:
            self.direction = -1
        elif commanded <= self.minimum:
            self.direction = 1
        return max(self.minimum, min(self.maximum, commanded + self.direction * self.step))


def orient_face_image(frame, clockwise_degrees):
    """Rotate only the detector input; keep raw RGB/depth coordinates unchanged."""
    if clockwise_degrees not in (0, 90, 180, 270):
        raise ValueError('Face image rotation must be 0, 90, 180, or 270 clockwise degrees')
    if frame is None or clockwise_degrees == 0:
        return frame
    import numpy as np
    return np.ascontiguousarray(np.rot90(frame, k=-(clockwise_degrees // 90)))
