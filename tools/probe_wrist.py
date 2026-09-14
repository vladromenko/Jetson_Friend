#!/usr/bin/env python3
"""Supervised one-shot J1/J2/J4 probe; no sweep, torque change, or automatic return.

--from-angle is the last commanded angle, NOT a measured joint position.
Only use while physically supervised and after verifying that posture is unchanged.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--j1-min', type=int, default=45)
    parser.add_argument('--j1-max', type=int, default=135)
    parser.add_argument('--from-angle', type=int, required=True)
    parser.add_argument('--to-angle', type=int, required=True)
    parser.add_argument('--output-prefix', required=True)
    args = parser.parse_args()
    max_step = 3 if args.joint == 1 else 5
    if not (0 <= args.from_angle <= 180 and 0 <= args.to_angle <= 180 and
            0 < abs(args.to_angle - args.from_angle) <= max_step):
        parser.error(f'Joint targets must be within 0..180 and differ by at most {max_step} degrees')
    if args.joint == 1 and not (0 <= args.j1_min <= args.from_angle <= args.j1_max <= 180
                               and args.j1_min <= args.to_angle <= args.j1_max):
        parser.error('J1 probe must remain inside the explicitly configured sector')
    import cv2
    import rclpy
    from robotics.manager import ArmCameraNode, ArmJoint
    from rclpy.duration import Duration
    lock = open(f"/tmp/milo-arm-controller-{os.getenv('ROS_DOMAIN_ID', '30')}.lock", 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rclpy.init()
    node = ArmCameraNode()
    try:
        end = time.monotonic() + 15
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.fresh() and node.joint_pub.get_subscription_count() > 0:
                break
        else:
            raise RuntimeError('Fresh RGB-D and arm subscriber required')
        for topic in (os.getenv('MILO_ARM_JOINT_TOPIC', '/arm_joint'),
                      os.getenv('MILO_ARM_JOINTS_TOPIC', '/arm6_joints')):
            if node.count_publishers(topic) != 1:
                raise RuntimeError(f'Competing arm publisher on {topic}; probe refused')
        frame, _ = node.snapshot()
        cv2.imwrite(args.output_prefix + '_before.jpg', frame)
        msg = ArmJoint()
        msg.id, msg.joint, msg.time = args.joint, args.to_angle, 2000
        print(f'J{args.joint} PROBE command {args.from_angle} -> {args.to_angle}, 2000 ms; no measured feedback', flush=True)
        sent = time.monotonic()
        node.joint_pub.publish(msg)
        acknowledged = node.joint_pub.wait_for_all_acked(Duration(seconds=2))
        print(f'DDS delivery acknowledgment={acknowledged}; not a servo-position acknowledgment', flush=True)
        if not acknowledged:
            raise RuntimeError('DDS acknowledgment timed out; physical outcome unknown; no retry')
        end = sent + 8
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.fresh():
            raise RuntimeError('Camera became stale after command; no more motion')
        frame, _ = node.snapshot()
        cv2.imwrite(args.output_prefix + '_after.jpg', frame)
        print('Probe finished; no further commands or automatic return', flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        lock.close()


if __name__ == '__main__':
    main()
