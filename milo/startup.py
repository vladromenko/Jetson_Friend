from __future__ import annotations


def step_toward(current: int, target: int, step_deg: int) -> int | None:
    """Return the next bounded startup target, or None when already close."""
    step = max(1, int(step_deg))
    current = int(current)
    target = int(target)
    if current == target:
        return None
    if current > target:
        return max(target, current - step)
    return min(target, current + step)
