"""Swappable perception interface; target-specific detection is not validated yet."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CandyDetection:
    label: str
    confidence: float
    bounding_box: tuple
    timestamp: float  # Image acquisition time, in ROS seconds.

    @property
    def center(self):
        x1,y1,x2,y2 = self.bounding_box
        return ((x1+x2)/2, (y1+y2)/2)


class CandyDetector(Protocol):
    def detect_candy(self, image, timestamp: float) -> CandyDetection | None:
        """Return a normalized raw-RGB box; never use a rotated face image here."""
        ...


class DisabledCandyDetector:
    def detect_candy(self, image, timestamp: float) -> CandyDetection | None:
        return None
