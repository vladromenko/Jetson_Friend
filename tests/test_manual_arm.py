import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / 'tools' / 'manual_arm.py'
spec = importlib.util.spec_from_file_location('manual_arm', module_path)
manual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manual)


def plan():
    return manual.ManualPlan({1:108,2:110,3:115,4:100,5:135,6:120})


def test_no_startup_motion():
    p=plan()
    assert p.next_ramp(0) is None


def test_ramp_reaches_70_in_bounded_steps_without_other_joints():
    p=plan();p.ramp=True;now=0
    original=dict(p.state)
    for _ in range(30):
        command=p.next_ramp(now)
        if command:
            joint,target=command
            assert joint == 2 and 70 <= target <= 110
            assert p.state[2] - target <= 2
            p.sent(joint,target,now)
        now+=2
    assert p.state[2] == 70 and not p.ramp
    assert all(p.state[j] == original[j] for j in (1,3,4,5,6))


def test_hold_cancels_ramp():
    p=plan();p.ramp=True;p.hold()
    assert p.next_ramp(10) is None


def test_moving_joint_does_not_queue_more_steps():
    p=plan();p.sent(1,110,0)
    assert p.request(1,2,.1) is None
    assert p.request(1,2,1.31) == (1,112)


def test_j1_cannot_cross_sector_boundary():
    p=plan();p.state[1]=134
    assert p.request(1,2,0) == (1,135)
    p.state[1]=45
    assert p.request(1,-2,0) is None


def test_pose_save_preserves_other_config_and_creates_backup(tmp_path):
    original='CAMERA_INDEX=0\nMILO_FACE_IMAGE_ROTATION_DEG=180\nMILO_ARM_FACE_SEARCH_POSE=90,115,115,110,135,120\n'
    (tmp_path/'config.env').write_text(original)
    p=plan();p.state[2]=70
    pose,backup=manual.save_pose(tmp_path,p.state)
    assert (backup/'config.env').read_text() == original
    text=(tmp_path/'config.env').read_text()
    assert 'CAMERA_INDEX=0' in text and 'MILO_FACE_IMAGE_ROTATION_DEG=180' in text
    assert 'MILO_ARM_FACE_SEARCH_POSE=108,70,115,100,135,120' in text
