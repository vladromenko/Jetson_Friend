#!/usr/bin/env python3
"""Supervised keyboard control; no arm commands at startup. ROS_DOMAIN_ID=30."""
import argparse
import curses
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class ManualPlan:
    """One small command at a time, with no queued keyboard movements."""
    def __init__(self, state):
        self.state = dict(state)
        self.ramp = False
        self.ready_at = 0.0
        self.runtime_ms = 1000

    def hold(self):
        self.ramp = False

    def request(self, joint, delta, now):
        if now < self.ready_at:
            return None
        limits = {1: (45, 135), 2: (70, 115)}
        if joint not in limits or abs(delta) > 2:
            raise ValueError('Manual control supports J1/J2 steps of at most 2 degrees')
        lo, hi = limits[joint]
        target = max(lo, min(hi, self.state[joint] + delta))
        if abs(target - self.state[joint]) > 2:
            raise ValueError('Declared commanded state is outside manual limits')
        if target == self.state[joint]:
            return None
        return joint, target

    def next_ramp(self, now):
        if not self.ramp:
            return None
        if self.state[2] <= 70:
            self.ramp = False
            return None
        return self.request(2, -min(2, self.state[2] - 70), now)

    def sent(self, joint, target, now):
        self.state[joint] = target
        self.ready_at = now + self.runtime_ms / 1000 + 0.30


def save_pose(root, state):
    """Only invoked by the operator's S key; back up exact config before editing."""
    import tempfile
    import shutil
    config = root / 'config.env'
    backups = root / 'data' / 'face_search_backups'
    backups.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%d_%H%M%S_manual_'), dir=backups))
    shutil.copy2(config, backup / 'config.env')
    key = 'MILO_ARM_FACE_SEARCH_POSE'
    lines = [line for line in config.read_text().splitlines() if not line.startswith(key + '=')]
    pose = ','.join(str(int(state[j])) for j in range(1, 7))
    lines.extend(['', '# Operator-selected commanded face-search pose; not measured feedback.', key + '=' + pose])
    config.write_text('\n'.join(lines) + '\n')
    return pose, backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commanded-state', required=True,
                        help='Last commanded J1,J2,J3,J4,J5,J6; never treated as position feedback')
    args = parser.parse_args()
    values = [int(x) for x in args.commanded_state.split(',')]
    if len(values) != 6 or not all(lo <= v <= hi for v, (lo, hi) in zip(
            values, ((45,135),(70,115),(0,180),(0,180),(0,270),(30,180)))):
        parser.error('Declared commanded state outside manual limits')
    root = Path(__file__).resolve().parents[1]
    root.joinpath('data').mkdir(exist_ok=True)
    lock = open(f"/tmp/milo-arm-controller-{os.getenv('ROS_DOMAIN_ID','30')}.lock", 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another MILO controller owns the arm; manual control refused')
    import rclpy
    from robotics.manager import ArmCameraNode, ArmJoint
    from rclpy.duration import Duration
    rclpy.init()
    node = ArmCameraNode()
    plan = ManualPlan({j+1: v for j,v in enumerate(values)})
    state_file = root / 'data' / 'manual_arm_commanded.json'
    blocked = False
    status = 'HOLD. No startup commands. Watch the robot and cable.'

    def record():
        state_file.write_text(json.dumps({'commanded': plan.state, 'measured': False,
            'updated': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, indent=2) + '\n')

    def run(screen):
        nonlocal status, blocked
        screen.nodelay(True)
        screen.keypad(True)
        curses.curs_set(0)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.025)
            now = time.monotonic()
            endpoint = node.joint_pub.get_subscription_count() > 0
            camera = node.fresh()
            exclusive = all(node.count_publishers(topic) == 1 for topic in
                (os.getenv('MILO_ARM_JOINT_TOPIC','/arm_joint'), os.getenv('MILO_ARM_JOINTS_TOPIC','/arm6_joints')))
            healthy = endpoint and camera and exclusive
            if not healthy:
                if plan.ramp:
                    blocked = True
                plan.hold()
            key = screen.getch()
            request = None
            if key in (ord('q'), ord('Q'), 27):
                plan.hold()
                break
            if key == ord(' '):
                plan.hold()
                curses.flushinp()
                status = 'HOLD: no new commands; the short in-flight move may finish.'
            elif key in (ord('r'), ord('R')):
                if healthy:
                    blocked = False
                    status = 'Readiness rechecked. HOLD; choose a movement explicitly.'
            elif key in (ord('g'), ord('G')) and healthy and not blocked:
                plan.ramp = True
                status = 'J2 -> 70 in 2-degree steps. SPACE holds immediately.'
            elif key in (ord('s'), ord('S')):
                plan.hold()
                if now >= plan.ready_at:
                    pose, backup = save_pose(root, plan.state)
                    status = f'Saved search pose {pose}; config backup created.'
                else:
                    status = 'Wait for current short move to finish, then press S.'
            elif key in (curses.KEY_LEFT, curses.KEY_RIGHT, curses.KEY_UP, curses.KEY_DOWN):
                plan.hold()
                if healthy and not blocked:
                    joint, delta = {curses.KEY_LEFT:(1,-2), curses.KEY_RIGHT:(1,2),
                                    curses.KEY_UP:(2,2), curses.KEY_DOWN:(2,-2)}[key]
                    request = plan.request(joint, delta, now)
                    # Key repeat never builds a queue of future motor commands.
                    curses.flushinp()
                else:
                    status = 'Motion blocked: check arm/camera/controller ownership. R rechecks.'
            if request is None and healthy and not blocked:
                request = plan.next_ramp(now)
            if request is not None:
                joint, target = request
                msg = ArmJoint()
                msg.id, msg.joint, msg.time = joint, target, plan.runtime_ms
                node.joint_pub.publish(msg)
                plan.sent(joint, target, time.monotonic())
                record()
                # Zero-time check keeps SPACE responsive. This is never position feedback.
                node.joint_pub.wait_for_all_acked(Duration(seconds=0))
                if not plan.ramp:
                    status = f'Commanded J{joint}={target}, {msg.time} ms. Not measured.'
            rows = [
                'MILO MANUAL ARM - operator supervision required',
                'COMMAND TARGETS ONLY - NO MEASURED JOINT FEEDBACK',
                '',
                '  '.join(f'J{j}={plan.state[j]:3d}' for j in range(1,7)),
                f'Arm subscriber={endpoint}   fresh RGB-D={camera}   exclusive={exclusive}',
                f'Mode: {"J2 RAMP TO 70" if plan.ramp else "HOLD / SINGLE STEP"}',
                '',
                'LEFT / RIGHT : J1 -2 / +2 deg (45..135)',
                'UP   / DOWN  : J2 +2 / -2 deg (70..115)',
                'G : slowly move J2 toward 70 deg; J1 remains fixed',
                'SPACE : stop new commands (keeps torque)',
                'S : save current commanded pose to FACE_SEARCH config',
                'R : recheck readiness after a fault; never starts motion',
                'Q / ESC / Ctrl+C : hold and exit',
                '',
                'J3/J4/J5/J6 are never commanded by this panel.',
                'One step: 1000 ms + 300 ms settling. No automatic movement at startup.',
                '', status,
            ]
            screen.erase()
            height,width=screen.getmaxyx()
            for i,line in enumerate(rows[:max(0,height-1)]):
                screen.addnstr(i,0,line,max(0,width-1))
            screen.refresh()
    try:
        curses.wrapper(run)
    except KeyboardInterrupt:
        pass
    finally:
        plan.hold()
        record()
        node.destroy_node()
        rclpy.shutdown()
        lock.close()
        print('Manual control closed. No further arm commands. Last commanded targets:', plan.state)


if __name__ == '__main__':
    main()
