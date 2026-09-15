"""Event-driven planning skeleton. It has no hardware executor."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time

STAGES = ('IDLE', 'FIND_CANDY', 'VALIDATE_CANDY', 'PLAN_PREGRASP', 'MOVE_PREGRASP',
          'MOVE_GRASP', 'CLOSE_GRIPPER', 'VERIFY_GRASP', 'LIFT', 'FIND_PERSON',
          'PLAN_HANDOVER', 'MOVE_HANDOVER', 'WAIT_FOR_HANDOVER', 'RELEASE', 'RETURN_HOME')
CONDITIONS = dict(zip(STAGES, ('start_requested', 'candy_detected', 'geometry_verified',
    'plan_valid', 'pregrasp_reached', 'grasp_reached', 'gripper_closed', 'grasp_verified',
    'lift_reached', 'person_detected', 'handover_plan_safe', 'handover_reached',
    'handover_confirmed', 'release_verified', 'home_reached')))
MOTION_STATES = {'MOVE_PREGRASP', 'MOVE_GRASP', 'CLOSE_GRIPPER', 'LIFT', 'MOVE_HANDOVER', 'RELEASE', 'RETURN_HOME'}


class ManipulationMachine:
    def __init__(self, clock=time.monotonic, timeout=15.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Timeout must be positive')
        self.clock, self.timeout = clock, timeout
        self.state, self.entered, self.failure = 'IDLE', clock(), None

    def tick(self, evidence=None, failure=None):
        evidence = evidence or {}
        if self.state == 'ERROR':
            return self.state
        if failure or (self.state != 'IDLE' and self.clock()-self.entered >= self.timeout):
            self.failure = failure or f'{self.state} timeout'
            self.state = 'ERROR'
        elif evidence.get(CONDITIONS[self.state]) is True:
            next_state = STAGES[(STAGES.index(self.state)+1) % len(STAGES)]
            if next_state in MOTION_STATES:
                self.state, self.failure = 'ERROR', 'Physical manipulation executor disabled pending calibration'
            else:
                self.state = next_state
            self.entered = self.clock()
        return self.state

    def reset(self, *, operator_verified_safe=False):
        if not operator_verified_safe:
            raise ValueError('Error recovery requires verified safe stationary state')
        self.state, self.entered, self.failure = 'IDLE', self.clock(), None


@dataclass(frozen=True)
class HandoverPolicy:
    head_clearance_m: float = 0.5
    body_clearance_m: float = 0.3
    confirmation_required: bool = True

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.head_clearance_m, self.body_clearance_m)):
            raise ValueError('Invalid safety distances')
        if not self.confirmation_required:
            raise ValueError('Handover requires explicit confirmation')

    def validate(self, *, source, head_distance, body_distance, confirmed):
        return (source in {'verified_hand', 'verified_handover_station'} and confirmed is True
                and all(math.isfinite(v) for v in (head_distance, body_distance))
                and head_distance >= self.head_clearance_m and body_distance >= self.body_clearance_m)


@dataclass
class AttemptRecord:
    timestamp: float
    attempt_id: str
    rgb_path: str | None = None
    depth_path: str | None = None
    observation: dict | None = None
    grasp_pose: dict | None = None
    joints: dict | None = None
    gripper_command: dict | None = None
    result: str = 'dry_run'
    failure_reason: str | None = None
    schema_version: int = 1
    metadata: dict = field(default_factory=dict)

    def append(self, path):
        payload = json.dumps(asdict(self), allow_nan=False)
        with Path(path).open('a', encoding='utf-8') as output:
            output.write(payload + '\n')
