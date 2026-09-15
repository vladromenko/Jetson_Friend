"""CameraInfo-based metric geometry. RGB/depth registration must be verified externally."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    frame_id: str
    k: tuple
    distortion: tuple
    model: str

    @classmethod
    def from_camera_info(cls, msg):
        if msg.binning_x not in (0, 1) or msg.binning_y not in (0, 1) or msg.roi.width or msg.roi.height:
            raise ValueError('Binned/cropped CameraInfo requires explicit adaptation')
        result = cls(msg.width, msg.height, msg.header.frame_id, tuple(msg.k), tuple(msg.d), msg.distortion_model)
        if not result.frame_id or len(result.k) != 9 or not all(math.isfinite(v) for v in result.k + result.distortion):
            raise ValueError('Invalid CameraInfo')
        if result.width <= 0 or result.height <= 0 or result.k[0] <= 0 or result.k[4] <= 0:
            raise ValueError('Uncalibrated CameraInfo')
        return result


def pixel_to_camera_point(u, v, depth_m, intrinsics):
    if not all(math.isfinite(x) for x in (u, v, depth_m)) or depth_m <= 0:
        raise ValueError('Invalid pixel/depth')
    if not (0 <= u < intrinsics.width and 0 <= v < intrinsics.height):
        raise ValueError('Pixel outside calibrated image')
    if intrinsics.model not in ('plumb_bob', 'rational_polynomial'):
        raise ValueError('Unsupported distortion model')
    k = np.asarray(intrinsics.k, dtype=np.float64).reshape(3, 3)
    d = np.asarray(intrinsics.distortion, dtype=np.float64)
    xy = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), k, d if d.size else None)[0, 0]
    return (float(xy[0]*depth_m), float(xy[1]*depth_m), float(depth_m))


def robust_depth(depth, u, v, encoding, radius=2, min_samples=8):
    if encoding not in ('16UC1', '32FC1'):
        raise ValueError('Unknown depth units')
    if depth.ndim != 2 or not all(math.isfinite(x) for x in (u, v)):
        raise ValueError('Invalid depth image/pixel')
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError('Depth pixel outside image')
    if not isinstance(radius, int) or not 1 <= radius <= 10 or min_samples < 1:
        raise ValueError('Invalid depth neighborhood')
    x, y = int(u), int(v)
    values = depth[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1].astype(float)
    values *= 0.001 if encoding == '16UC1' else 1.0
    values = values[np.isfinite(values) & (values >= 0.12) & (values <= 2.0)]
    if values.size < min_samples:
        raise ValueError('Insufficient valid depth')
    median = float(np.median(values))
    if float(np.percentile(values, 90)-np.percentile(values, 10)) > max(0.03, median*0.05):
        raise ValueError('Depth discontinuity: object/background ambiguous')
    return median


@dataclass(frozen=True)
class CandyObservation:
    timestamp: float
    pixel_position: tuple
    depth: float
    point_camera: tuple
    frame_id: str
    confidence: float
    label: str = 'candy'


def observe_candy(box, confidence, depth, encoding, intrinsics, *, color_shape,
                  color_frame, depth_frame, color_stamp, depth_stamp, now,
                  registration_verified=False):
    if not registration_verified:
        raise ValueError('RGB/depth registration requires validation')
    if depth.shape != (intrinsics.height, intrinsics.width) or tuple(color_shape[:2]) != depth.shape:
        raise ValueError('RGB/depth resolution mismatch')
    if color_frame != intrinsics.frame_id or depth_frame != color_frame:
        raise ValueError('RGB/depth optical frame mismatch')
    if not all(math.isfinite(t) and 0 <= now-t <= 0.75 for t in (color_stamp, depth_stamp)) or abs(color_stamp-depth_stamp) > 0.05:
        raise ValueError('Stale or unsynchronized RGB/depth')
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid confidence')
    if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box) or not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError('Invalid bounding box')
    u, v = (box[0]+box[2])*0.5*(intrinsics.width-1), (box[1]+box[3])*0.5*(intrinsics.height-1)
    z = robust_depth(depth, u, v, encoding)
    return CandyObservation(color_stamp, (u,v), z, pixel_to_camera_point(u,v,z,intrinsics), intrinsics.frame_id, confidence)


def transform_observation(observation, buffer, target_frame, *, calibration_verified=False):
    if not calibration_verified or not target_frame:
        raise ValueError('Camera-to-arm calibration requires validation')
    from geometry_msgs.msg import PointStamped
    from rclpy.time import Time
    from rclpy.duration import Duration
    import tf2_geometry_msgs  # Registers PointStamped support with TF2.
    msg = PointStamped()
    msg.header.frame_id = observation.frame_id
    msg.header.stamp = Time(seconds=observation.timestamp).to_msg()
    msg.point.x, msg.point.y, msg.point.z = observation.point_camera
    return buffer.transform(msg, target_frame, timeout=Duration(seconds=0.1))
