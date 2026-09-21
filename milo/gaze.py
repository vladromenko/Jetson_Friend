from __future__ import annotations


def camera_to_gaze(cx: float, cy: float, x_sign: int, y_sign: int) -> tuple[float, float]:
    x = x_sign * (cx - 0.5) * 2.0
    y = y_sign * (cy - 0.5) * 2.0
    return max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y))
