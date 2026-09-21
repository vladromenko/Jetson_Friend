#!/usr/bin/env python3
"""Raise the screen using one feedback-confirmed joint at a time."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from milo.config import Config
from milo.ros_runtime import RosRuntime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=int, choices=(2, 3), default=2)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--j2-correction", action="store_true")
    args = parser.parse_args()
    limits = (90, 165) if args.joint == 2 else (0, 180)
    if not limits[0] <= args.target <= limits[1]:
        parser.error(f"target must be within {limits}")

    ros = RosRuntime(Config.load(Path(__file__).resolve().parent))
    try:
        ros.start()
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if (ros.node and ros.node.pub.get_subscription_count() > 0
                    and time.monotonic() - ros.feedback_stamp < 1.0
                    and args.joint in ros.latest_feedback):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"fresh J{args.joint} feedback or STM32 subscriber unavailable")

        for _ in range(20):
            current = int(ros.latest_feedback[args.joint])
            print(f"[J{args.joint}] measured={current}, target={args.target}", flush=True)
            if abs(current - args.target) <= 2:
                return 0
            correcting_j2 = args.joint == 2 and args.j2_correction and current < args.target
            if correcting_j2 and args.target - current > 5:
                raise RuntimeError("J2 correction is limited to 5 degrees per run")
            if (args.joint == 2 and current < args.target and not correcting_j2) or (args.joint == 3 and current > args.target):
                raise RuntimeError("refusing a movement in the unverified direction")
            next_angle = (min(args.target, current + 5) if correcting_j2 else
                          max(args.target, current - 5) if args.joint == 2
                          else min(args.target, current + 5))
            ok, reason = ros.command_aux_joint(args.joint, next_angle, 1500, limits, 10.0)
            if not ok:
                raise RuntimeError(reason)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and ros.arm_busy():
                time.sleep(0.05)
            measured = int(ros.latest_feedback.get(args.joint, current))
            if abs(measured - args.target) <= 2:
                print(f"[J{args.joint}] reached within 2 degrees: {measured}", flush=True)
                return 0
            moved = (measured - current if correcting_j2 or args.joint == 3 else current - measured)
            if moved < 2 or abs(measured - next_angle) > 3:
                raise RuntimeError(f"J{args.joint} did not follow command: {current} -> {measured}")
        raise RuntimeError(f"J{args.joint} target was not reached within 20 steps")
    finally:
        ros.close()


if __name__ == "__main__":
    raise SystemExit(main())
