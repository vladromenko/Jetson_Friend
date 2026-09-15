#!/usr/bin/env python3
"""Read-only camera/face/geometry probe. Creates no arm publishers."""
import argparse
from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120:
        parser.error('--seconds must be 1..120')
    os.environ['MILO_ARM_ENABLE'] = '0'
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    from robotics.manager import RobotManager
    from robotics.geometry import CameraIntrinsics, robust_depth, pixel_to_camera_point
    robot = RobotManager()
    rclpy.init()
    node = rclpy.create_node('milo_manipulation_dry_run')
    bridge = CvBridge()
    latest, counts, infos = {}, {}, {}
    stamps = {'color': deque(maxlen=100), 'depth': deque(maxlen=100)}
    def receive(msg, key):
        latest[key] = (msg, time.monotonic())
        counts[key] = counts.get(key, 0) + 1
        stamps[key].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    for key, env, default in [('color', 'MILO_ARM_COLOR_TOPIC', '/camera/color/image_raw'),
                              ('depth', 'MILO_ARM_DEPTH_TOPIC', '/camera/depth/image_raw')]:
        node.create_subscription(Image, os.getenv(env, default), lambda msg, k=key: receive(msg,k), qos_profile_sensor_data)
        info_topic = os.getenv('MILO_ARM_'+key.upper()+'_INFO_TOPIC', default.replace('image_raw', 'camera_info'))
        node.create_subscription(CameraInfo, info_topic, lambda msg, k=key: infos.update({k: msg}), qos_profile_sensor_data)
    report = {'arm_publishers_created': 0, 'detector': robot.face_detector[0] if robot.face_detector else None,
              'detections': 0, 'tracked_frames': 0, 'proposals': 0, 'detector_ms': [], 'samples': []}
    start, cpu_start = time.monotonic(), time.process_time()
    last_frame = None
    next_log = start
    last_logged_state = None
    try:
        while time.monotonic()-start < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.05)
            if 'color' in latest:
                msg, arrival = latest['color']
                if arrival != last_frame:
                    last_frame = arrival
                    frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
                    begin = time.monotonic()
                    boxes = robot._raw_faces(frame)
                    report['detector_ms'].append((time.monotonic()-begin)*1000)
                    target = robot.target_tracker.update([(b, robot.face_scores.get(b)) for b in boxes], arrival, time.monotonic())
                    report['detections'] += len(boxes)
                    report['tracked_frames'] += target.state == 'tracked'
                    report['proposals'] += bool(target.correction_deg)
                    state_changed = target.state != last_logged_state
                    if target.correction_deg or state_changed or time.monotonic() >= next_log:
                        item = asdict(target)
                        item['event'] = 'proposal' if target.correction_deg else ('state_change' if state_changed else 'heartbeat')
                        item['proposals_total'] = report['proposals']
                        item['target_x'] = robot.target_tracker.target_x
                        item['box_near_image_edge'] = target.box is not None and (
                            min(target.box[:2]) < 0.01 or max(target.box[2:]) > 0.99)
                        item['pixel_error_from_center'] = None if target.center is None else [
                            (target.center[0]-0.5)*frame.shape[1], (target.center[1]-0.5)*frame.shape[0]]
                        print(json.dumps(item), flush=True)
                        report['samples'].append(item)
                        last_logged_state = target.state
                        next_log = time.monotonic()+1
        elapsed = time.monotonic()-start
        report['elapsed_s'] = elapsed
        report['cpu_core_percent'] = 100*(time.process_time()-cpu_start)/elapsed
        report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['received_fps'] = {k: v/elapsed for k,v in counts.items()}
        report['processed_fps'] = len(report['detector_ms'])/elapsed
        report['streams'] = {k: {'width': m.width, 'height': m.height, 'encoding': m.encoding,
                                 'frame': m.header.frame_id} for k,(m,_) in latest.items()}
        report['intrinsics'] = {k: asdict(CameraIntrinsics.from_camera_info(m)) for k,m in infos.items()}
        report['registration_verified'] = False
        report['arm_transform_valid'] = False
        if stamps['color'] and stamps['depth']:
            report['nearest_stamp_skew_s'] = min(abs(stamps['color'][-1]-d) for d in stamps['depth'])
        if 'depth' in latest and 'depth' in infos:
            msg, _ = latest['depth']
            intr = CameraIntrinsics.from_camera_info(infos['depth'])
            depth = bridge.imgmsg_to_cv2(msg, 'passthrough')
            try:
                u,v = intr.width//2, intr.height//2
                z = robust_depth(depth,u,v,msg.encoding)
                report['depth_center_geometry_test'] = {'pixel': [u,v], 'depth_m': z,
                    'point_camera': pixel_to_camera_point(u,v,z,intr), 'frame': intr.frame_id,
                    'meaning': 'depth-only stationary scene sample, not a candy observation'}
            except ValueError as exc:
                report['depth_center_geometry_test'] = {'rejected': str(exc)}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in {'samples','detector_ms'}}, indent=2))
        return 0 if len(latest) == 2 and len(infos) == 2 and report['detector_ms'] else 1
    finally:
        robot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
