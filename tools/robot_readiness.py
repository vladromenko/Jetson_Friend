#!/usr/bin/env python3
"""Read-only ROS endpoint/stream readiness probe. Never publishes arm commands."""
import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['arm', 'camera'])
    parser.add_argument('--timeout', type=float, default=45)
    args = parser.parse_args()
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.type_support import check_for_type_support
    from arm_msgs.msg import ArmJoint, ArmJoints
    from std_msgs.msg import Int32
    from sensor_msgs.msg import Image
    for typ in (ArmJoint, ArmJoints, Int32):
        check_for_type_support(typ)
    rclpy.init()
    node = rclpy.create_node('milo_readiness_probe')
    seen = {}
    pubs = []
    if args.kind == 'arm':
        for typ, key, default in ((ArmJoint, 'MILO_ARM_JOINT_TOPIC', '/arm_joint'),
                                  (ArmJoints, 'MILO_ARM_JOINTS_TOPIC', '/arm6_joints'),
                                  (Int32, 'MILO_ARM_TORQUE_TOPIC', '/arm_torque')):
            pubs.append(node.create_publisher(typ, os.getenv(key, default), 1))
    else:
        for key, default in (('MILO_ARM_COLOR_TOPIC', '/camera/color/image_raw'),
                             ('MILO_ARM_DEPTH_TOPIC', '/camera/depth/image_raw')):
            node.create_subscription(Image, os.getenv(key, default),
                lambda msg, k=key: seen.update({k: time.monotonic()}) if msg.data else None,
                qos_profile_sensor_data)
    end = time.monotonic() + args.timeout
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.kind == 'arm':
                ready = all(p.get_subscription_count() > 0 for p in pubs)
            else:
                ready = len(seen) == 2 and all(time.monotonic() - t < 0.75 for t in seen.values())
            if ready:
                print(f'[MILO] {args.kind} readiness verified')
                return 0
        print(f'[MILO] {args.kind} readiness timed out after {args.timeout:g}s')
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
