# MILO face-search implementation and test report

Date: 2026-09-15. Software implemented and deployed; physical face acquisition remains unverified.

## Findings and root causes

- Mac and Jetson tracked source started at `main`, commit `7165481`; manager and robot launcher hashes matched. Mac initially had a clean working tree. No applicable AGENTS.md was found.
- The old manager performed automatic J1/J4 sign probes, used an OpenCV object tracker, and could substitute a stale face lock for a current detection during calibration. Those mechanisms did not establish trustworthy feedback or direction. They have been removed.
- Startup checked camera freshness but did not require arm subscribers. The launcher checked topic names, waited arbitrary short sleeps, and printed a warning without disabling arm motion.
- `_sleep` read the clock twice around its deadline, allowing a negative sleep argument.
- The launcher searched broadly for an agent, could select different executables, and did not enforce exclusive serial ownership. Its final `exec` also bypassed normal EXIT cleanup. During diagnostic validation, stopping a ROS launch parent left its camera child running; launcher-owned processes now use separate process groups and group cleanup. The orphan from that diagnostic was stopped explicitly.
- SIGINT/SIGTERM previously stopped audio/vision before arm shutdown. Signals now inhibit new arm commands immediately.

## Changes

- `src/robotics/face_search.py`: independently testable detector-only acquisition, three matching distinct consecutive frames, 2.5-second loss timeout, bounded J1 stepping and reversal, validated finite configuration.
- `src/robotics/manager.py`: FACE_SEARCH_POSE → FACE_SEARCH → FACE_FOUND. Freeze during acquisition and transient loss; clear expired locks. No continuous J2/J3/J4 tracking. Fresh RGB-D and all three matched arm publishers are required. Connectivity loss latches SAFE. Process and thread locks prevent competing MILO motion owners. Candy implementation is retained, but voice candy dispatch is disabled for this milestone. ROS absence no longer prevents importing the rest of MILO. Sensor subscriptions accept sensor-data QoS. Joint state is explicitly commanded state.
- `start_milo_robot.sh`: deterministic agent path, stable CP2104 path, ownership inspection, launcher/UART locks, bounded endpoint and actual image readiness, latched startup inhibition, `--no-arm-motion` diagnostic mode, owned process-group cleanup.
- `tools/robot_readiness.py`, `tools/robot_serial_owner.py`: read-only readiness and serial ownership probes.
- `start.sh`: preserve launcher inhibition across config reload.
- `src/main.py`: inhibit arm immediately on shutdown signal.
- `config.env.example`: search pose, sector, timing, confirmation and loss settings.
- `tests/test_face_search.py`: 26 robotics safety tests; 61 tests total in the repository.

Initial test configuration: pose `90,115,115,110,135,120`, pose duration 5000 ms; J1 sector 45–135°, step 3°, runtime 320 ms, settle 0.10 s, initial increasing direction. This is a test pose, not a calibrated pose. Optional `MILO_FACE_SEARCH_TEST_DURATION_SEC=20` capped this supervised run; the normal default is 0 (no test timer). On test expiry the final implementation requires restarting before another test.

There is no verified position-hold/cancel command or measured joint feedback. FACE_FOUND prevents new motion commands; a short command already sent may finish. Emergency voice stop retains the existing torque-off behavior. Ordinary shutdown/test expiry inhibits further commands without automatically releasing torque.

## Jetson environment and backup

- Active SSH target: `vlad@192.168.0.177`; hostname `vlad`; repository `/home/vlad/Jetson_Friend`.
- Configured `jetson` (`vlados@192.168.1.19`) and `rosmaster` (`jetson@192.168.0.168`) timed out. Recent MILO copy history identified the working configured `.177` target.
- Linux aarch64, kernel `6.8.12-1021-tegra`; Python 3.12.3; ROS Jazzy; domain 30.
- Environment: `/opt/ros/jazzy/setup.bash`, `/home/vlad/m3pro_hw_ws/install/setup.bash`, and repository `orbbec_ws/install/setup.bash`.
- `ArmJoint` and `ArmJoints` Python type support verified.
- Installed agent verified at `/home/vlad/microros_ws/install/micro_ros_agent/lib/micro_ros_agent/micro_ros_agent`.
- Existing agent PID 7862 uses `/home/vlad/microros_ws/build/micro_ros_agent/micro_ros_agent`; reused without restarting it.
- Serial: `/dev/serial/by-id/usb-Silicon_Labs_CP2104_USB_to_UART_Bridge_Controller_02E0E664-if00-port0` → `/dev/ttyUSB0`, 2,000,000 baud.
- Backup: `/home/vlad/Jetson_Friend/data/face_search_backups/20260915_013058.EKmozV/`. Includes manager, launchers, main, config.env, and config.env.example before their respective edits. The first deployment preserved config.env; a later timestamped backup preceded the detector-rotation setting described below.

## Live tests and evidence

1. No-motion full MILO launch: reached `MILO ready`, loaded the local model stack, and logged explicit arm inhibition. All arm/color/depth readiness checks succeeded.
2. DaBai: color 640×480, depth 640×400, both fresh; Haar detection roughly 30–55 ms per frame in an independent read-only probe. Depth/color pixel alignment was not calibrated or used for manipulation.
3. Arm: `/YB_Example_Node`; `/arm_joint`, `/arm6_joints`, `/arm_torque` present with matched subscribers. Exactly one micro-ROS agent throughout.
4. Supervised movement test, authorized by the user: software logged the five-second pose command, J1 targets 93…135, reversal at 135, then targets down to 108, followed by the 20-second test stop. No focus calibration or Python exception appeared. These are commanded targets, not verified physical positions.
5. No FACE_FOUND occurred in the live test. The saved post-test frame shows monitor/wall, appears rolled, and contains no visible face. Mount orientation/pose require user observation before calibration or another movement test.
6. A second launcher was rejected with exit 1. A deliberately nonexistent arm topic caused the readiness probe to time out with exit 1; the real arm probe then succeeded.
7. Both Mac and Jetson: `pytest -q` → 61 passed; Python compilation, shell syntax, and `git diff --check` passed. Deployed hashes match all nine modified/new executable/config/test files. Tests explicitly cover no commands after FACE_FOUND, transient loss, timeout/reacquisition, distinct images, stale images, endpoint/camera loss, emergency stop, bounded reversal, invalid settings, and test deadline.
8. One initial manual import probe failed because that probe replaced PYTHONPATH with `src`, discarding ROS paths. Repeating with the sourced environment and adding src to sys.path passed. This was a probe setup error, not a deployed ROS import failure.

Runtime logs and camera image are in Jetson `data/face_search_diagnostic.log`, `data/face_search_supervised.log`, and `data/face_search_camera_after_test.jpg`; local copies are under ignored `artifacts/face_search/`.

## Remaining issues and final state

- Physical height and initial movement were subsequently confirmed by the user; see the wrist investigation below. Physical face acquisition and a live FACE_FOUND freeze/loss/reacquisition cycle remain unverified. Do not infer motion success from ROS/logs.
- Logitech: `/dev/v4l/by-id` absent during discovery; `/dev/video0` cannot open. Separate from working DaBai. No global camera configuration was changed.
- TensorRT: existing cross-device engine-plan and logger warnings were reproduced. No engine/model rebuild or deletion was performed.
- The full MILO test and its camera were stopped. A separate camera-only session was subsequently started for supervised wrist probes; see below. The original agent remains running. Firmware, STM32 protection and models were not modified.
- Git: no commit or push. Five tracked files modified, four implementation/test files added, plus this report. Jetson pre-existing `deps/llama.cpp`, `Log/`, `log.txt`, and `src/robotics/manager.py.before_face_search` changes/files preserved.


## Wrist orientation investigation (subsequent supervised work)

User observation is ground truth: height is approximately correct; the initial camera aim is backward; increasing J1 moves toward the user. J2/J3 and J1 sweep direction therefore remain unchanged. No J1 sector expansion has been made.

The initially installed M3 chassis URDF did not describe this arm. The manufacturer archive at `Downloads/ROSMASTER M3 PRO-Code/jetson nano°Ґpi/M3Pro_ws-0.0.2.zip`, member `m3pro_ws/src/M3Pro/urdf/M3Pro.urdf`, does: `DCW2_Joint` is fixed to parent `arm4`, while `arm5_Joint` connects arm4 to arm5. For this standard mounting, J4 controls camera aim and J5 does not move the camera. The manufacturer Robotic arm solution PDF page 4 documents the all-90-degree upright reference; no example motion from the PDF was executed.

Added `tools/probe_wrist.py`: one-shot supervised J4-only command, maximum 5-degree change within firmware limits, 2000 ms duration, required fresh RGB-D/subscriber, controller lock and competing-publisher rejection, before/after captures, no torque change, no automatic return. Its `--from-angle` is declared last commanded state, not measured feedback.

- Probe 110 → 105: user confirmed motion **toward them**, cable clear. All other joints fixed at last commanded J1=108, J2=115, J3=115, J5=135, J6=120.
- Probe 105 → 100: completed; physical feedback pending. Camera features shifted mostly vertically (median raw image displacement approximately -1.3, -48.5 pixels, 35 RANSAC inliers). First probe displacement was approximately -0.9, -33.4 pixels, 37 inliers. These image changes are relative observations, not a calibrated kinematic measurement or user-direction inference.
- Latest commanded state: `108,115,115,100,135,120`. No face visible in saved probe image. No autonomous sweep running.

Image orientation was evaluated independently at 0/90/180/270 degrees. A 180-degree rotation makes monitor text/dock upright; quarter-turn alternatives remain sideways. Added `MILO_FACE_IMAGE_ROTATION_DEG`, applied only to Haar/detector input, preserving raw RGB/depth pixel coordinates. Default example value is 0; Jetson value is 180. The prior config is at `data/face_search_backups/20260915_014551.IjnMew/config.env`. Five additional rotation tests passed; repository total is 61 on both machines. Physical aiming remains necessary despite this image correction.

Camera-only ROS launch PID 16820 was started for these probes. It owns no arm motion loop. The original micro-ROS agent remains PID 7862. Remaining next step: physical report for J4=100 before another small wrist step, then a bounded search retest once aim is suitable.
