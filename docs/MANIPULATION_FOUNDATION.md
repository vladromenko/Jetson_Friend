# MILO manipulation foundation — 2026-09-15

## Status

The first software increment is deployed and tested on the Jetson. Live YuNet detection,
smoothed tracking, dry-run corrections and depth-camera projection work. Physical tracking,
RGB candy localization in the arm frame, grasping and handover are **not validated**.
No arm or torque commands were sent during this work. All temporary camera/agent/manager
processes were stopped; the machine was returned to its initially idle ROS state.

## Discovered environment and preserved work

- SSH: `vlad@192.168.0.177`, hostname `vlad`. Configured `jetson` and `rosmaster` timed out.
- Ubuntu 24.04.4, aarch64, kernel 6.8.12-1021-tegra, Orin GPU, 7,485 MiB reported RAM.
- ROS 2 Jazzy, domain 30; CUDA compiler 13.2.86, driver 595.78, Docker 29.8.0.
- Project `/home/vlad/Jetson_Friend`; Python 3.12 virtual environment.
- Mac was clean at `198afd0`; fast-forwarded to Jetson's `e90aa6e` before editing.
- Jetson's uncommitted `tools/manual_arm.py`, model/dependency work, patches and backups
  were preserved. Its manual-arm diff hash was checked unchanged. No commit or push.
- Modified pre-existing files backed up under `data/manipulation_audit/backup/` on Jetson.
- Other inspected trees: `m3pro_hw_ws`, `microros_ws`, project `orbbec_ws`,
  `yahboom_m3pro/yahboomm3_ws`, standalone face-servo/calibrator payloads. Standalone
  controllers use different limits and automatic calibration; they were not run.

## Actual ROS interfaces

The graph was initially idle. Starting the existing camera driver and micro-ROS agent
without a motion publisher exposed:

| Interface | Type / owner | Meaning |
|---|---|---|
| `/arm_joint` | `arm_msgs/msg/ArmJoint`, subscriber `/YB_Example_Node` | `uint8 id`, `int16 joint`, `int16 time` |
| `/arm6_joints` | `arm_msgs/msg/ArmJoints`, same node | six int16 joint values and int16 time |
| `/arm_torque` | `std_msgs/msg/Int32`, same node | torque control; untouched |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image`, `/camera/camera` | RGB8, 640×480 |
| `/camera/depth/image_raw` | same | 16UC1, 640×400, millimetres |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | RGB calibration |
| `/camera/depth/camera_info` | same | depth calibration |
| `/camera/depth/points` | `sensor_msgs/msg/PointCloud2` | driver point cloud, not validated for grasping |
| `/camera/depth_to_color` | `orbbec_camera_msgs/msg/Extrinsics` | factory camera extrinsics |
| `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` | camera frames only |

The arm node has **no publishers, services or actions**: no measured joint positions,
acknowledged completion, force feedback or verified cancel primitive. It offers six
logical channels; the project treats J6 as the gripper, so six channels do not establish
six independent positioning DOFs plus a gripper. Degrees and milliseconds follow existing
project/controller usage; physical zero, mechanical travel and calibration remain unverified.

Project logical bounds retained: J1–J4 0–180°, J5 0–270°, J6 30–180°.
These are not certified mechanical/collision limits. Existing config says gripper open=120,
closed=55; these endpoints were not exercised. Home and historical face poses are retained
as configuration, never automatically commanded. No firmware source was located/changed.

UART: CP2104 serial `02E0E664`, stable `/dev/serial/by-id/...02E0E664-if00-port0`,
2,000,000 baud. Existing executable:
`/home/vlad/microros_ws/install/micro_ros_agent/lib/micro_ros_agent/micro_ros_agent`.

The installed arm_msgs colcon package hook omitted its ament prefix, so Python imported
messages while `ros2 interface show` reported Unknown package. Explicitly sourcing
`~/m3pro_hw_ws/install/arm_msgs/share/arm_msgs/local_setup.bash` fixes discovery;
both launchers now use this hook. No rebuild was necessary.

## Camera geometry and TF

DaBai DCW2, serial CH82C5200KB, firmware RD1014, Orbbec SDK 1.10.37, wrapper 1.5.22,
USB 2.0, nominal 10 FPS. Launch parameters report `align_mode=HW`, target `COLOR`,
frame synchronization **false**. Output image sizes, frames and intrinsics differ;
these RGB/depth streams must not be treated as registered.

Observed values (documentation only; code consumes live CameraInfo):

| Stream | fx | fy | cx | cy | Optical frame |
|---|---:|---:|---:|---:|---|
| RGB | 480.707855 | 480.711273 | 322.290588 | 245.641464 | camera_color_optical_frame |
| Depth | 305.863556 | 305.863556 | 322.264893 | 198.329956 | camera_depth_optical_frame |

Both advertise rational_polynomial distortion; RGB coefficients are nonzero. Projection
uses OpenCV undistortion before multiplying the camera ray by metric depth. Neighborhood
median rejects zeros, NaNs, out-of-range depth, insufficient samples and strong discontinuities.
Supported depth encodings are explicit: 16UC1 millimetres and 32FC1 metres, never magnitude guesses.

`view_frames` found camera_link → camera_depth_frame → color/IR frames and their optical
frames. No robot/arm base, arm links or end-effector transform is being published. Available
ROSMASTER-M3 Xacro describes the chassis, not a verified arm model. No running FK, IK,
MoveIt or joint-state publisher was found. Old reports of a manufacturer arm archive do
not establish the current machine's arm geometry.

`observe_candy` rejects unverified registration, mismatched sizes/frames, >50 ms skew and
>750 ms age. `transform_observation` requires explicit calibration verification, a named
target frame and timestamped TF2 lookup; missing TF raises instead of inventing an identity.
Camera-on-arm mounting will require validated joint-dependent transforms, not a fixed
camera-to-base transform that becomes wrong when the arm moves.

## Implementation

- `src/robotics/manager.py`: integrates tracking, CameraInfo and guarded arm interface;
  no automatic initial pose or blind sweep; voice cannot bypass physical enable gates.
- `safety.py`: rejects invalid IDs, NaN/Inf, bounds violations, unknown start state,
  >1° steps or >3°/s commands. Validates rounded values actually sent to firmware.
- `face_target.py`: existing FaceSearch association/three-frame confirmation plus exponential
  center smoothing (alpha .25), confidence filtering, deadband, ≤2 one-degree proposals/s,
  stable local track IDs and lost-target handling. Camera callbacks reject stale/replayed
  source timestamps before refreshing freshness. Every missed/stale frame produces zero
  correction; association expires after 2.5 seconds. IDs are tracks, not person identities.
- `yunet_compat.py`: OpenCV 4.6 FaceDetectorYN accepted the ONNX model but failed at inference.
  Added warmup validation and a DNN adapter reusing the existing `vision.TensorRTYuNet`
  decoding logic. Existing TensorRT/global vision and SFace identity code remain intact.
  The compatibility path uses the existing small ONNX model on CPU; no new model downloaded.
  Haar fallback has unknown confidence and cannot produce physical correction proposals.
- `geometry.py`: intrinsics, robust depth, projection, CandyObservation and gated TF2 adapter.
- `candy.py`: replaceable detector protocol and disabled default. Existing Gemma box helper
  remains available, but has not been validated on candy. No target detector is claimed working.
- `manipulation.py`: explicit states, per-state success evidence, timeouts/error latch,
  manual safe reset, handover policy and versioned lightweight JSONL attempt schema.
  A transition into any motion state fails closed: no manipulation executor is installed.
- `tools/manipulation_dry_run.py`, `start_manipulation_dry_run.sh`: read-only camera diagnostic,
  target logs, timing/memory counters and depth projection. The probe creates no arm publishers.
- `start_milo_robot.sh`, `config.env.example`: discovery hook, startup messaging and enable gates.
- `tests/test_face_search.py`, `tests/test_manipulation.py`: updated no-sweep expectation,
  command gates, projection, tracking, handover and state-machine tests.

Arm methods: `move_joints`, `get_joint_state`, `stop`; Cartesian moves, home, gripper endpoints
and end-effector feedback explicitly refuse or return unavailable until verified.
`stop` inhibits new commands; it cannot cancel a command already sent. It deliberately does
not release torque automatically because this can drop the arm/load. Physical power-stop
behavior still needs hardware verification. Competing ROS publishers inhibit motion.

Face coordinates refer to the configured rotated detector image. Candy pixels remain in raw
RGB coordinates. Current Jetson settings are rotation=0, horizontal target=.36 and J1 sign=+1;
they were preserved, not revalidated. The old code's unconditional sign-verification claim
was removed. Handover must not use face center as a target; policy defaults are 0.5 m head
and 0.3 m body clearance plus confirmation, all provisional and not a collision model.

## Running and physical gates

Read-only test on Jetson, from the repository:

```sh
./start_manipulation_dry_run.sh 30
```

Starts only the camera if necessary, reuses an existing fresh stream, records
`data/manipulation_audit/live.json`, and cleans up only its own camera process.
For the whole existing application: `./start_milo_robot.sh --no-arm-motion`.
The full AI/audio application was not relaunched in this increment.

Physical tracking is OFF by default, even with existing `MILO_ARM_ENABLE=1`.
Enabling requires all of: `MILO_ARM_TRACKING_ENABLED=1`,
`MILO_ARM_CONVENTIONS_VERIFIED=1`, and `MILO_ARM_VERIFIED_CURRENT_POSE` containing six
operator-verified current logical angles. Do not copy an old home pose into this field.
The pose is a human declaration, not feedback. J1-only corrections use ≥500 ms duration;
physical sessions stop after 20 seconds by default. Connectivity loss/test expiry requires
restart and fresh pose verification. No automatic torque enable, home move or grasp occurs.

## Actual validation and evidence

- Mac and Jetson: 96 tests passed, Python compilation and diff checks passed.
- Live probe initially failed with OpenCV `getLayerData` error. Fixed and rerun; failure log
  retained as `data/manipulation_audit/live.log`, successful log `live_fixed.log`.
- Successful 20 s run: 182 detections, 173 tracked frames, 33 dry-run proposals;
  RGB processed 9.35 FPS, depth received 9.50 FPS including discovery time.
  Detector median 42.73 ms, p95 48.27 ms; CPU ~160% of one core, peak RSS ~206 MiB.
  Camera container snapshot ~10.3% CPU and 118 MiB RSS. GPU utilization was not available
  from nvidia-smi; this compatibility inference path is CPU. No heavyweight model loaded.
- Latest observed RGB/depth timestamp skew 28.4 ms. This is one sample, not a synchronization
  guarantee; the geometry API checks every observation.
- First center-depth sample was rejected for a depth discontinuity. Subsequent camera-only
  launcher test obtained pixel (320,200), depth 1.262 m, camera XYZ
  (-0.009345, 0.006891, 1.262) m in camera_depth_optical_frame. This is a scene-depth
  software check, not a measured candy position or validated robot coordinate.
- Integrated manager ran 8 s with fresh images and both CameraInfo messages. ROS monitoring
  counted joint=0, pose=0, torque=0 commands; armed=false. No face was visible to its detector
  during that separate run; lost state remained safe.
- Camera-only launcher startup/probe/cleanup tested. Arm ROS subscribers and message fields
  inspected; no actual joint motion, gripper closure or torque operation tested.
- Jetson `data/manipulation_audit/`: backups, camera/agent logs, graph inventories, TF graph,
  `live_20sec.json`, `manager_live.log`, `launcher_test.log`. Local evidence copies under
  ignored `artifacts/manipulation/`.

## Engineering TODO / next milestone

1. Human visual check: run the 30 s diagnostic; keep the robot stationary, bring the entire
   face into the camera image, move slowly left/right, leave view for >3 s and re-enter.
   Expect stable tracked state/ID while visible, bounded proposals, lost/zero corrections
   when absent, and **no physical motion**. Report whether detections correspond to you,
   any cropped face, and whether any motion occurred. Detection logs alone do not establish
   ground-truth face correctness or movement direction.
2. Verify current pose, J1 sign and available travel physically. Then one supervised 1° J1
   step from that verified pose (≥500 ms), with cable clearance and stop access, before
   enabling any timed tracking. This physical prerequisite was not guessed remotely.
3. Resolve RGB/depth registration and timestamp synchronization; validate XYZ with a
   stationary measured object at several positions. Validate camera mounting and arm TF.
4. Obtain correct arm description, joint conventions and feedback/cancel capabilities;
   verify FK before choosing lightweight IK or MoveIt. No large stack installed yet.
5. Add controlled-object candy detector and planner; exercise the state machine in simulation.
6. Next manipulation milestone: verified stationary-object camera→arm coordinates and
   collision-checked dry-run HOME/PRE_GRASP/GRASP/LIFT waypoints, then a supervised fixed-object
   grasp. Human handover follows separate clearance, hand detection and release validation.

## Follow-up: interpreting diagnostic output

The user's subsequent log kept track ID 1 with confidence approximately .92, but its
bounding box touched the image's top edge. This suggests cropping; human visual confirmation
is still needed, and a stationary box alone does not prove it follows the intended person.
The initial two-second readiness timeout precedes camera startup and is expected when idle.
The OpenCV incompatibility message is followed by a successful compatibility decoder startup.

The diagnostic previously printed only one sampled frame per second, which could repeatedly
miss the brief rate-limited correction proposals between samples. It now prints every nonzero
proposal and every state transition, plus a heartbeat, cumulative proposal count, target_x and
an image-edge flag. A heartbeat's zero correction does not mean there were no proposals in
that interval. No motion policy or physical enable setting changed in this logging fix.
