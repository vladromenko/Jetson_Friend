# MILO runtime architecture and distributed proposal

> Historical design document. The proposal described here was subsequently
> implemented and extended in [the distributed MILO repository](https://github.com/vladromenko/MILO).
> The verified Jetson-only baseline remains accurate for this repository.

Date: 2026-09-21

This document describes the verified Jetson baseline before any Raspberry Pi + Hailo work. The distributed design is a proposal only; the RC13 behavior remains unchanged.

## Verified baseline

The canonical application package on the Jetson is `milo/` in `/home/vlad/Jetson_Friend`. Every Python module and test matches the archived RC13 source byte-for-byte.

The work history leading to RC13 was also checked:

- RC11 removed a false direction check that could permanently pause tracking when the person moved.
- RC12 added vertical following, corrected the cat eye direction, and retained the J2/J6 command prohibition.
- RC13 increased tracking speed while retaining feedback gating, joint limits, and slower motion near the target.

The low-level micro-ROS agent was rebuilt after a broken workspace relocation. A standalone hardware check confirmed `/arm_joint`, `/arm_torque`, `/arm6_feedback`, `/arm6_joints`, and `/arm6_raw`; a real six-joint feedback packet was received without sending a motion command.

## Actual startup graph

```text
manual command: ./start_milo.sh
  |
  +-- load config.env and lock data/milo.lock
  +-- apply display rotation
  +-- source ROS 2 Jazzy and ros_ws/install
  +-- start micro_ros_agent over CP2104 UART
  +-- start Orbbec DaBai ROS launch
  +-- wait for color and depth topics
  +-- start llama-server with Gemma + mmproj
  +-- preflight: text, vision, audio, camera, face detector, arm feedback
  `-- python -m milo.main
        +-- ROS executor thread
        +-- camera/face/tracking thread
        +-- proactive observation thread
        +-- voice interaction loop
        `-- pygame cat display
```

`start_milo.sh` owns every process it starts and stops the complete process group on `Ctrl+C`, TERM, or launcher failure. A file lock prevents two MILO instances.

## Runtime execution flow

```text
Milo.start()
  -> FaceUI.start()
  -> RosRuntime.start()
  -> startup_lift(), only when enabled and safe feedback is fresh
  -> wait for a real DaBai frame
  -> vision loop
       latest_color -> detect_face -> gaze -> FaceFollower.plan
       -> ArmSafetyGate -> /arm_joint
  -> proactive loop
       fresh frame -> VLM observation -> optional spoken observation
  -> voice loop
       microphone -> VAD -> Whisper -> intent routing
       -> Gemma text or VLM -> memory update -> Piper -> speaker
```

## Module ownership

| Module | Responsibility | Direct dependencies |
| --- | --- | --- |
| `milo/main.py` | Lifecycle and voice/vision/proactive orchestration | Every application subsystem |
| `milo/config.py` | Environment parsing and validated configuration | `config.env` |
| `milo/ros_runtime.py` | ROS node, camera subscriptions, arm feedback and commands | rclpy, sensor_msgs, arm_msgs |
| `milo/safety.py` | Per-joint limits, feedback stability and command gating | Arm feedback |
| `milo/startup.py` | Bounded startup pose stepping | Safety gate and arm transport |
| `milo/tracking.py` | Face error to safe J1/J3 tracking intent | Face observations and joint state |
| `milo/vision.py` | YuNet face detection and image encoding | OpenCV, optional TensorRT engine |
| `milo/audio.py` | VAD, Whisper STT, Piper TTS, device selection | sounddevice, scipy, binaries |
| `milo/llm.py` | Local text/VLM client | llama-server HTTP API |
| `milo/memory.py` | Persistent last-seen object locations | Local JSON data |
| `milo/intents.py` | Visual and object-location intent routing | Pure Python |
| `milo/gaze.py` | Camera coordinates to cat-eye coordinates | Pure Python |
| `milo/ui.py`, `cat_face.py` | User-facing animated display | pygame and X11 display |
| `milo/preflight.py` | End-to-end dependency checks before main | Selected runtime subsystems |

## Hardware and external dependencies

```text
ROS 2 Jazzy (/opt/ros/jazzy)
  `-- ros_ws/install
       +-- arm_msgs (tracked custom source)
       +-- micro_ros_msgs (pinned upstream commit)
       +-- micro_ros_agent (pinned upstream commit)
       `-- OrbbecSDK_ROS2 (pinned upstream commit)

CP2104 serial @ 2,000,000 baud
  <-> STM32 micro-ROS firmware
      +-- /arm_joint command
      +-- /arm_torque command
      +-- /arm6_feedback measured angles
      +-- /arm6_joints state
      `-- /arm6_raw diagnostic state

Orbbec DaBai
  -> /camera/color/image_raw
  -> /camera/depth/image_raw
  `-- compressed color fallback

Local AI
  +-- llama.cpp + Gemma multimodal GGUF + mmproj
  +-- whisper.cpp + base.en model
  +-- Piper + en_US Ryan voice
  `-- YuNet ONNX face detector

USB/UI
  +-- Audio-Technica UM02 input
  +-- UACDemo output
  `-- X11 display on DP-1
```

## Safety invariants to preserve

- No command is sent from stale camera or arm feedback.
- Commands are absolute, bounded, and feedback-gated.
- One unsafe joint does not disable safe axes, but it cannot be commanded.
- J2 is only used by the explicitly configured startup lift; J6 is never commanded.
- Tracking must remain functional when the TensorRT face engine is absent by using ONNX/Haar fallback.
- Loss of Raspberry Pi connectivity must stop body commands without stopping Jetson conversation or memory.

## Minimal-change Jetson/Pi design

Do not split `milo.main` or convert the application into many ROS nodes. Introduce one body boundary and preserve current call sites through a local adapter.

```text
MILO application on Jetson
  -> RobotBody interface
       +-- LocalRobotBody (current ros_runtime behavior)
       `-- RosRobotBodyClient
             <network ROS 2>
             -> Raspberry Pi Body Server
                  +-- DaBai acquisition
                  +-- Hailo face/person/object inference
                  +-- CP2104 arm transport
                  +-- feedback normalization and watchdog
                  `-- optional display/audio transport later
```

The first interface should expose only stable body facts and commands:

```python
class RobotBody:
    def latest_frame(self): ...
    def latest_detections(self): ...
    def joint_state(self): ...
    def command_joint(self, joint_id, angle, runtime_ms): ...
    def health(self): ...
```

`LocalRobotBody` delegates to the existing `RosRuntime`; therefore the baseline continues to work during the migration.

## Placement proposal

| Capability | Initial owner | Distributed target | Reason |
| --- | --- | --- | --- |
| Dialogue, policy, memory | Jetson | Jetson | Stateful high-level behavior |
| Gemma text/VLM | Jetson | Jetson | Existing model runtime and RAM |
| Whisper/Piper logic | Jetson | Jetson initially | Avoid changing a working voice loop |
| Cat expression policy | Jetson | Jetson | Part of social behavior |
| Display rendering | Jetson | Pi only if display cable moves | Physical placement decision |
| RGB-D acquisition | Jetson | Raspberry Pi | Camera belongs with the body |
| Face/person/object detection | Jetson CPU/OpenCV | Hailo on Pi | Low-latency body perception |
| Tracking decision | Jetson initially | Split: target on Jetson, servo loop on Pi | Network-safe control |
| micro-ROS serial bridge | Jetson | Raspberry Pi | Keep arm link physically local |
| Hard joint limits/watchdog | Jetson | Both Jetson and Pi | Defense in depth |

## Migration sequence

1. Freeze and manually verify this RC13 baseline. Tag the verified commit only after the hardware test.
2. Create a new feature branch for distributed work; never develop it on the baseline branch.
3. Add `RobotBody` plus `LocalRobotBody` with characterization tests. Runtime behavior must remain identical.
4. Add body health/heartbeat messages and a Pi process that publishes no hardware commands yet.
5. Move DaBai acquisition to Pi. Jetson consumes frames through the body interface; keep local fallback.
6. Move face/person detection to Hailo and transmit compact detections, not continuous full-resolution frames, for tracking.
7. Move the CP2104 micro-ROS agent and low-level command watchdog to Pi. Preserve the existing message types and limits.
8. Decide separately whether display and audio hardware move. They are not prerequisites for the camera/arm split.
9. Remove the local fallback only after repeated cold-boot, disconnect, latency, and emergency-stop tests.

Each stage must keep a runnable local mode and have an explicit rollback commit. The Raspberry Pi work begins only after the user confirms the baseline camera, arm, display, speech, vision, and interaction behavior.
