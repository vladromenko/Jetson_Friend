# Integrated MILO arm control — current supervised validation

## Why corrections previously did not move the robot

`start_manipulation_dry_run.sh` deliberately creates no arm publishers. Its nonzero
corrections are diagnostic only. Production already instantiated `RobotManager` in
`src/main.py`; its motion gate additionally required physical enable, verified conventions
and a verified current pose. Those requirements were not supplied by the diagnostic launch.

There is no new ROS protocol or parallel runtime. The production path is:

`start_milo_robot.sh → start.sh → src/main.py → RobotManager → move_joint_relative /
move_joints → ArmCameraNode publishers → micro-ROS agent → /YB_Example_Node → STM32`.

The final firmware-to-servo implementation is opaque: no firmware source or measured
joint feedback is available. Existing manual control uses exactly these same publishers.
`/arm_joint` uses `arm_msgs/msg/ArmJoint`: id (uint8 channel), joint (int16 absolute logical
angle in degrees), time (int16 milliseconds). `/arm6_joints` uses `arm_msgs/msg/ArmJoints`
with six absolute targets and a duration. J6 is the gripper channel in existing software.

## One owner and state

`RobotManager` is the authoritative high-level controller within MILO, owning the ROS
executor, worker and tracking lifecycle. Face following calls its relative joint method;
future manipulation must use that same interface. MANUAL, FACE_TRACK and DISABLED control
modes arbitrate motion. MANIPULATION is reserved and currently refuses physical commands.
Existing process locks coordinate the separate manual tool; competing topic publishers
also inhibit commands. No modifications were made to the user's manual-arm implementation.

Commanded state updates only after publish, independently from any claim of measurement.
It is initialized from explicitly supplied current angles or from a one-shot operator-approved
absolute pose command. No old manual-state file is automatically trusted. Normal corrections
are <=1 degree with >=500 ms interval, and FACE_TRACK is J1-only with <=15 degree excursion
from its session origin, additionally bounded by the configured J1 sector.

An explicit initialization may move all six channels over 5000 ms. This is authorized only
by `--arm-initialize-pose`; it does not run from a default/home guess. It uses the same
publisher path and consumes its one-shot request. Torque is not automatically toggled.
DDS acknowledgment proves transport receipt, not actual servo position or successful grasp.
Missing acknowledgment inhibits subsequent commands and requires physical state verification.

## Normal-runtime commands

No movement:

```sh
./start_milo_robot.sh --no-arm-motion
```

The exact supervised Test B executed after the user authorized the initial pose:

```sh
./start_milo_robot.sh --arm-j1-probe --arm-initialize-pose 67,91,95,0,91,118
```

This starts the complete MILO process, initializes once over 5 seconds, commands J1 67→68→67
with 1000 ms per step, and disables further arm motion. AI, speech and the rest of MILO
remain the ordinary application; this is not a standalone arm controller.

When the CURRENT pose is physically confirmed, skip any initial pose move by using
`--arm-current-pose` instead. Do not reuse old angles after manually moving/reinitializing
servos. Face following has a normal-runtime `--arm-face-track` switch, requires an explicit
experimentally verified `--arm-j1-sign`, and uses `--arm-target-x` (default geometric center
0.5) and `--arm-track-seconds` (default 30, maximum 120). Physical face-follow testing must
wait for the successful physical Test B and sign verification. The prior 0.36 target is
an inherited setting claimed calibrated in an earlier commit, without enough current
measurement evidence to adopt blindly; the CLI exposes the target explicitly.

## Evidence so far

- Connected to `vlad@192.168.0.177`; current source hashes compared before editing.
- Normal complete MILO run reached `MILO ready`, DaBai streams and face tracking active.
  Logitech `/dev/video0` remains unavailable independently of the working arm camera.
- `/YB_Example_Node` subscribes reliably to all three arm topics, with no publishers,
  services or actions. Existing CP2104 2,000,000-baud agent was reused through the launcher.
- Test B normal runtime emitted:
  1. `/arm6_joints`: 67,91,95,0,91,118, 5000 ms, DDS ack=true.
  2. `/arm_joint`: id=1, joint=68, time=1000, DDS ack=true.
  3. `/arm_joint`: id=1, joint=67, time=1000, DDS ack=true.
- During Test B, `/arm_joint` had one publisher (`milo_arm_camera`) and one subscriber
  (`YB_Example_Node`), both RELIABLE/VOLATILE.
- Physical motion confirmation is pending from the operator. Do not label face following
  physically verified until a human observes the response.
- 97 tests passed on Mac and Jetson after the integrated probe change. These tests do not
  establish physical movement. Jetson logs: `data/integrated_arm/test_a.log`, `test_b.log`;
  pre-change backups: `data/integrated_arm/backup.OWQlC2/`.

Next: operator confirms initial pose and J1 out/return, then experimentally verify image
correction sign and perform at least 30 seconds of integrated face following, including
reversal, lost-target stop and excursion/oscillation monitoring. Candy work is unchanged.
