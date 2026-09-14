import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robotics.face_search import FaceSearch
from robotics import manager

BOX = (0.3, 0.2, 0.5, 0.5)


def test_confirmation_requires_distinct_consecutive_images():
    search = FaceSearch()
    for _ in range(10):
        assert search.observe([BOX], 1, 1) == 'FACE_SEARCH'
    assert search.hits == 1
    search.observe([], 2, 2)
    assert search.hits == 0
    for stamp in (3, 4):
        assert search.observe([BOX], stamp, stamp) == 'FACE_SEARCH'
    assert search.observe([BOX], 5, 5) == 'FACE_FOUND'


def test_loss_freezes_then_clears_stale_identity():
    search = FaceSearch()
    for stamp in (0.1, 0.2, 0.3):
        search.observe([BOX], stamp, stamp)
    assert search.observe([], 1, 1) == 'FACE_FOUND'
    assert search.observe([], 2.7, 2.7) == 'FACE_FOUND'
    assert search.observe([], 2.9, 2.9) == 'FACE_SEARCH'
    assert search.box is None and search.hits == 0


def test_different_face_does_not_refresh_lock():
    search = FaceSearch()
    for stamp in (0.1, 0.2, 0.3):
        search.observe([BOX], stamp, stamp)
    other = (0.8, 0.7, 0.95, 0.95)
    search.observe([other], 1, 1)
    assert search.last_seen == 0.3
    assert search.observe([other], 3, 3) == 'FACE_SEARCH'
    assert search.hits == 1


def test_many_sweeps_remain_bounded_and_reverse():
    search = FaceSearch()
    angle = 90
    angles = []
    for _ in range(500):
        target = search.next_target(angle)
        assert 45 <= target <= 135
        assert abs(target - angle) <= 3
        angles.append(target)
        angle = target
    assert min(angles) == 45 and max(angles) == 135
    with pytest.raises(ValueError):
        search.next_target(0)


@pytest.mark.parametrize('kwargs', [dict(step=90), dict(step=float('nan')),
    dict(minimum=-1), dict(maximum=360), dict(runtime_ms=20), dict(direction=0),
    dict(required_hits=1), dict(loss_timeout=0)])
def test_reject_unsafe_configuration(kwargs):
    with pytest.raises(ValueError):
        FaceSearch(**kwargs)


@pytest.fixture
def robot(monkeypatch):
    monkeypatch.setenv('MILO_ARM_ENABLE', '0')
    r = manager.RobotManager()
    r.enabled = r.armed = r.face_follow = r.pose_ready = True
    r.mode = 'FACE_SEARCH'
    pub = lambda: SimpleNamespace(get_subscription_count=lambda: 1, publish=Mock())
    r.node = SimpleNamespace(joint_pub=pub(), joints_pub=pub(), torque_pub=pub(),
                             fresh=lambda: True, face_snapshot=lambda: (object(), time.monotonic()))
    r.face_detector = object()
    r._raw_faces = Mock(return_value=[])
    monkeypatch.setattr(manager, 'ArmJoint', SimpleNamespace, raising=False)
    return r


def test_manager_found_produces_no_commands_then_resumes_j1_only(robot):
    robot._raw_faces.return_value = [BOX]
    for _ in range(20):
        robot._face_tick()
    assert robot.mode == 'FACE_FOUND'
    robot.node.joint_pub.publish.assert_not_called()
    robot.node.joints_pub.publish.assert_not_called()
    robot._raw_faces.return_value = []
    robot._face_tick()
    robot.node.joint_pub.publish.assert_not_called()
    robot.face_search.last_seen -= 3
    robot._face_tick()
    assert robot.mode == 'FACE_SEARCH'
    msg = robot.node.joint_pub.publish.call_args.args[0]
    assert (msg.id, msg.joint, msg.time) == (1, 93, 320)
    robot.node.joints_pub.publish.assert_not_called()


def test_missing_endpoint_latches_safe(robot):
    robot.node.joint_pub.get_subscription_count = lambda: 0
    robot._face_tick()
    assert robot.mode == 'SAFE' and not robot.armed
    robot.node.joint_pub.get_subscription_count = lambda: 1
    robot._face_tick()
    robot.node.joint_pub.publish.assert_not_called()


def test_camera_loss_and_emergency_stop_prevent_commands(robot):
    robot.node.fresh = lambda: False
    robot._face_tick()
    assert robot.mode == 'SAFE'
    robot.node.fresh = lambda: True
    robot.armed = True
    robot.mode = 'FACE_SEARCH'
    robot.stop_request.set()
    robot._face_tick()
    robot.node.joint_pub.publish.assert_not_called()


def test_startup_block_cannot_be_bypassed_by_voice(robot):
    robot.startup_block = 'readiness timed out'
    for command in ('arm ready', 'follow my face', 'find candy'):
        robot.handle_voice(command)
    assert robot.jobs.empty()
    assert not robot._can_move()
    robot.node.joint_pub.publish.assert_not_called()


def test_sleep_never_waits_negative(robot, monkeypatch):
    samples = iter([10.0, 10.01, 10.2])
    monkeypatch.setattr(manager.time, 'monotonic', lambda: next(samples))
    robot.stop_request = Mock(is_set=lambda: False)
    assert robot._sleep(0.1)
    assert all(0 <= call.args[0] <= 0.05 for call in robot.stop_request.wait.call_args_list)


def test_sleep_stop_interrupts(robot):
    robot.stop_request = threading.Event()
    robot.stop_request.set()
    assert not robot._sleep(100)


def test_test_deadline_stops_search(robot):
    robot.test_deadline = time.monotonic() - 1
    robot._face_tick()
    assert robot.mode == 'STOPPED' and not robot.armed
    assert robot.stop_request.is_set()
    robot.node.joint_pub.publish.assert_not_called()


def test_stale_processed_image_cannot_move(robot):
    robot.node.face_snapshot = lambda: (object(), time.monotonic() - 2)
    robot._face_tick()
    robot.node.joint_pub.publish.assert_not_called()


def test_startup_does_not_clear_emergency_stop(robot):
    robot.stop_request.set()
    robot.armed = False
    robot._startup_sequence()
    assert robot.stop_request.is_set() and not robot.armed
    robot.node.joint_pub.publish.assert_not_called()


@pytest.mark.parametrize('degrees,expected', [(0, [[1,2,3],[4,5,6]]),
    (90, [[4,1],[5,2],[6,3]]), (180, [[6,5,4],[3,2,1]]), (270, [[3,6],[2,5],[1,4]])])
def test_face_rotation_preserves_raw_image(degrees, expected):
    import numpy as np
    from robotics.face_search import orient_face_image
    raw = np.array([[1,2,3],[4,5,6]])
    assert orient_face_image(raw, degrees).tolist() == expected
    assert raw.tolist() == [[1,2,3],[4,5,6]]


def test_face_rotation_rejects_unconfigured_angle():
    from robotics.face_search import orient_face_image
    with pytest.raises(ValueError):
        orient_face_image(None, 45)


@pytest.mark.parametrize('args', [
    ['--joint','1','--from-angle','135','--to-angle','138'],
    ['--joint','1','--from-angle','108','--to-angle','112'],
    ['--joint','4','--from-angle','100','--to-angle','90'],
    ['--joint','4','--from-angle','180','--to-angle','181'],
    ['--joint','5','--from-angle','135','--to-angle','138'],
])
def test_manual_probe_rejects_unsafe_input_before_ros(args):
    import subprocess
    import sys
    from pathlib import Path
    probe = Path(__file__).resolve().parents[1] / 'tools' / 'probe_wrist.py'
    result = subprocess.run([sys.executable, str(probe), *args, '--output-prefix', '/tmp/milo_refused_probe'],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'error:' in result.stderr
