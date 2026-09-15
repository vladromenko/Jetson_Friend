from types import SimpleNamespace
from unittest.mock import Mock
import math
import numpy as np
import pytest
from robotics.safety import ArmSafety
from robotics.face_target import FaceTargetTracker
from robotics.geometry import CameraIntrinsics, robust_depth, pixel_to_camera_point, observe_candy, transform_observation
from robotics.manipulation import ManipulationMachine, HandoverPolicy, AttemptRecord


def intrinsics():
    # Synthetic calibration, used only to test geometry.
    return CameraIntrinsics.from_camera_info(SimpleNamespace(width=20, height=20,
        header=SimpleNamespace(frame_id='optical'), k=[100,0,10,0,100,10,0,0,1],
        d=[0]*5, distortion_model='plumb_bob', binning_x=0, binning_y=0,
        roi=SimpleNamespace(width=0,height=0)))


@pytest.mark.parametrize('target,duration', [(math.nan,500),(math.inf,500),(181,500),(95,500),(91,320),(91,math.nan),(91,6000)])
def test_motion_rejects_nonfinite_range_step_and_speed(target,duration):
    with pytest.raises(ValueError):
        ArmSafety({1:(0,180)}).validate({1:target},{1:90},duration)


def test_motion_requires_known_start_and_returns_actual_integer_command():
    safety = ArmSafety({1:(0,180)})
    assert safety.validate({1:90.8},{1:90},500) == {1:91}
    with pytest.raises(ValueError):
        safety.validate({1:91},{},500)


def test_tracker_smooths_limits_rate_and_stops_on_lost_stale_low_score():
    t = FaceTargetTracker()
    box = (.6,.3,.8,.6)
    for now in (.1,.2,.3):
        target = t.update([(box,.9)],now,now)
    assert target.state == 'tracked' and target.correction_deg == 1
    target = t.update([((.65,.3,.85,.6),.9)],.4,.4)
    assert .70 < target.center[0] < .75 and target.correction_deg == 0
    assert t.update([], .5,.5).correction_deg == 0
    assert t.update([(box,.9)], .4,.9).correction_deg == 0
    assert t.update([(box,.1)], 1,1).correction_deg == 0
    assert t.update([],4,4).box is None
    new = t.update([(box,.9)],4.1,4.1)
    assert new.track_id == 2 and new.state == 'acquiring'


def test_haar_unknown_confidence_never_proposes_motion():
    t = FaceTargetTracker()
    for now in (.1,.2,.3,1):
        result = t.update([((.6,.3,.8,.6),None)],now,now)
    assert result.state == 'tracked' and result.correction_deg == 0


def test_projection_depth_units_and_invalid_values():
    intr = intrinsics()
    assert pixel_to_camera_point(10,10,.5,intr) == (0,0,.5)
    assert pixel_to_camera_point(12,10,.5,intr) == pytest.approx((.01,0,.5))
    depth = np.full((20,20),500,dtype=np.uint16)
    depth[10,10] = 0
    assert robust_depth(depth,10,10,'16UC1') == .5
    assert robust_depth(depth.astype(float)/1000,10,10,'32FC1') == .5
    depth[8:11,8:13] = 1500
    with pytest.raises(ValueError, match='discontinuity'):
        robust_depth(depth,10,10,'16UC1')
    with pytest.raises(ValueError):
        pixel_to_camera_point(-1,10,.5,intr)
    with pytest.raises(ValueError):
        robust_depth(np.zeros((20,20)),10,10,'32FC1')


def observation(**changes):
    args = dict(color_shape=(20,20,3),color_frame='optical',depth_frame='optical',
        color_stamp=1,depth_stamp=1.01,now=1.1,registration_verified=True)
    args.update(changes)
    return observe_candy((.4,.4,.6,.6),.9,np.full((20,20),500), '16UC1',intrinsics(),**args)


@pytest.mark.parametrize('kwargs', [dict(registration_verified=False),dict(color_shape=(24,20,3)),
    dict(depth_frame='other'),dict(depth_stamp=.5),dict(now=3),dict(color_stamp=math.nan)])
def test_registration_sync_and_frame_gates(kwargs):
    with pytest.raises(ValueError):
        observation(**kwargs)


def test_transform_fails_closed_before_tf_or_ros_import():
    buffer = Mock()
    with pytest.raises(ValueError):
        transform_observation(observation(),buffer,'arm')
    buffer.transform.assert_not_called()


def test_state_machine_timeout_disabled_executor_and_manual_recovery():
    now = [0]
    m = ManipulationMachine(clock=lambda: now[0])
    assert m.tick({'start_requested':True}) == 'FIND_CANDY'
    now[0]=16
    assert m.tick() == 'ERROR'
    with pytest.raises(ValueError):
        m.reset()
    m.reset(operator_verified_safe=True)
    for evidence in ('start_requested','candy_detected','geometry_verified'):
        m.tick({evidence:True})
    assert m.state == 'PLAN_PREGRASP'
    assert m.tick({'plan_valid':True}) == 'ERROR'
    assert 'disabled' in m.failure


def test_handover_rejects_face_target_and_requires_confirmation():
    p = HandoverPolicy()
    assert not p.validate(source='face',head_distance=1,body_distance=1,confirmed=True)
    assert not p.validate(source='verified_hand',head_distance=1,body_distance=1,confirmed=False)
    assert not p.validate(source='verified_hand',head_distance=.2,body_distance=1,confirmed=True)


def test_attempt_record(tmp_path):
    import json
    path = tmp_path/'attempts.jsonl'
    AttemptRecord(1,'test',failure_reason='uncalibrated').append(path)
    assert json.loads(path.read_text())['schema_version'] == 1


def test_manager_physical_gate_and_stop(monkeypatch):
    from robotics import manager
    import time
    monkeypatch.setenv('MILO_ARM_ENABLE','0')
    r = manager.RobotManager()
    pub = lambda: SimpleNamespace(get_subscription_count=lambda:1,publish=Mock())
    r.node = SimpleNamespace(joint_pub=pub(),joints_pub=pub(),torque_pub=pub(),
        fresh=lambda:True,face_snapshot=lambda:(object(),time.monotonic()))
    r.enabled = r.armed = r.face_follow = True
    monkeypatch.setattr(manager,'ArmJoint',SimpleNamespace,raising=False)
    monkeypatch.setattr(r,'_sleep',lambda _:True)
    assert not r.move_joints({1:91},500)
    r.physical_tracking = r.conventions_verified = True
    r.control_mode = "FACE_TRACK"
    r.verified_pose = dict(r.state)
    assert r.move_joints({1:91},500)
    assert r.node.joint_pub.publish.call_args.args[0].joint == 91
    with pytest.raises(ValueError):
        r.move_joints({1:100},500)
    count = r.node.joint_pub.publish.call_count
    r.stop()
    assert not r.move_joints({1:92},500)
    assert r.node.joint_pub.publish.call_count == count
    r.node.torque_pub.publish.assert_not_called()


def test_manager_proposal_publishes_only_verified_one_degree(monkeypatch):
    from robotics import manager
    import time
    monkeypatch.setenv('MILO_ARM_ENABLE','0')
    r = manager.RobotManager()
    pub = lambda: SimpleNamespace(get_subscription_count=lambda:1,publish=Mock())
    r.node = SimpleNamespace(joint_pub=pub(),joints_pub=pub(),torque_pub=pub(),
        fresh=lambda:True,face_snapshot=lambda:(object(),time.monotonic()))
    r.enabled = r.armed = r.face_follow = r.physical_tracking = r.conventions_verified = True
    r.control_mode = "FACE_TRACK"
    r.verified_pose = dict(r.state)
    r.face_scores = {(.6,.3,.8,.6):.9}
    r._raw_faces = Mock(return_value=[(.6,.3,.8,.6)])
    monkeypatch.setattr(manager,'ArmJoint',SimpleNamespace,raising=False)
    monkeypatch.setattr(r,'_sleep',lambda _:True)
    for _ in range(6):
        r._face_tick()
    assert r.node.joint_pub.publish.call_count == 1
    msg = r.node.joint_pub.publish.call_args.args[0]
    assert (msg.id,msg.joint,msg.time) == (1,91,500)
    r.node.joints_pub.publish.assert_not_called()
    assert r.test_deadline is None  # Continuous face tracking has no automatic test timeout.


def test_camera_replayed_and_old_header_cannot_refresh_image():
    from robotics.manager import ArmCameraNode
    import threading
    node = SimpleNamespace(lock=threading.Lock(), color_header=None, color=None, color_time=0,
        bridge=SimpleNamespace(imgmsg_to_cv2=lambda *a,**kw:np.zeros((2,2,3))),
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=10_000_000_000)))
    node._image_recent = lambda msg:ArmCameraNode._image_recent(node,msg)
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=9,nanosec=900_000_000)))
    ArmCameraNode._color_cb(node,msg)
    first = node.color_time
    assert first > 0
    ArmCameraNode._color_cb(node,msg)
    assert node.color_time == first
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=8,nanosec=0)))
    ArmCameraNode._color_cb(node,msg)
    assert node.color_time == first


def test_integrated_pose_then_probe_uses_same_publisher_path(monkeypatch, tmp_path):
    from robotics import manager
    monkeypatch.setenv('MILO_ARM_ENABLE','0')
    monkeypatch.setenv('MILO_ARM_INITIALIZE_POSE','67,91,95,0,91,118')
    monkeypatch.setenv('MILO_ARM_PROBE_J1','1')
    monkeypatch.setenv('MILO_ARM_MODE','MANUAL')
    r = manager.RobotManager()
    r.enabled = r.physical_tracking = True
    pub = lambda:SimpleNamespace(get_subscription_count=lambda:1,publish=Mock())
    r.node=SimpleNamespace(joint_pub=pub(),joints_pub=pub(),torque_pub=pub(),fresh=lambda:True)
    monkeypatch.setattr(manager,'ArmJoint',SimpleNamespace,raising=False)
    monkeypatch.setattr(manager,'ArmJoints',SimpleNamespace,raising=False)
    now=[100.0]
    monkeypatch.setattr(manager.time,'monotonic',lambda:now[0])
    def wait(seconds):
        now[0]+=seconds
        return True
    monkeypatch.setattr(r,'_sleep',wait)
    assert r.state == {}
    assert r.initialize_pose(r.initial_pose_request)
    initial=r.node.joints_pub.publish.call_args.args[0]
    assert (initial.joint1,initial.joint4,initial.time)==(67,0,5000)
    r.armed=True
    r.root=tmp_path
    monkeypatch.setattr(r,'_capture_probe_frame',lambda *_:None)
    r.run_j1_probe()
    assert [(c.args[0].id,c.args[0].joint,c.args[0].time) for c in r.node.joint_pub.publish.call_args_list] == [(1,68,1000),(1,67,1000)]
    assert r.state == {1:67,2:91,3:95,4:0,5:91,6:118}
    assert r.control_mode == 'DISABLED' and not r.armed
    r.node.torque_pub.publish.assert_not_called()
    with pytest.raises(RuntimeError):
        r.initialize_pose(r.state)
