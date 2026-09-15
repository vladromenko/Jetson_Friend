"""Validate logical degree commands; firmware limits are not collision limits."""
import math


class ArmSafety:
    def __init__(self, limits, max_speed=3.0, max_step=1.0):
        self.limits = limits
        self.max_speed = max_speed
        self.max_step = max_step
        if not all(math.isfinite(v) and v > 0 for v in (max_speed, max_step)):
            raise ValueError('Invalid motion bounds')

    def validate(self, targets, current, duration_ms):
        if not targets or not math.isfinite(duration_ms) or not 320 <= duration_ms <= 5000:
            raise ValueError('Duration must be finite and 320..5000 ms')
        if int(duration_ms) != duration_ms:
            raise ValueError('Duration must be whole milliseconds')
        result = {}
        for joint, value in targets.items():
            if isinstance(joint, bool) or joint not in self.limits or joint not in current:
                raise ValueError('Unknown joint or unverified starting pose')
            lo, hi = self.limits[joint]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'J{joint}: invalid angle')
            prior = current[joint]
            if not math.isfinite(prior) or not lo <= prior <= hi:
                raise ValueError('Invalid starting pose')
            rounded = int(round(value))
            delta = abs(rounded - prior)
            if delta > self.max_step or delta / (duration_ms / 1000) > self.max_speed:
                raise ValueError('Command exceeds step or speed bound')
            result[joint] = rounded
        return result
