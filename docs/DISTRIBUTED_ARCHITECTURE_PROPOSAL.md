# MILO Runtime Architecture And Distributed Migration Proposal

Date: 2026-09-21
Branch: `distributed-jetson-pi-architecture`

This document records what is actually running on the Jetson before any Raspberry Pi + Hailo split is attempted. The goal is to preserve the working implementation first, then introduce distribution behind narrow interfaces.

## Executive Summary

The live Jetson is not running the local Mac repository layout directly. The live robot starts `/home/vlad/MILO/start_milo.sh`, where `/home/vlad/MILO` is a symlink to `/home/vlad/MILO_CLEAN_RC13`. That runtime package is named `milo/`.

The Mac repository at `/Users/vladromenko/Desktop/Scoltech/Jetson friend` is currently on branch `distributed-jetson-pi-architecture`, but its application package is `src/`, not the live `milo/` package. This means the latest working implementation is not represented one-to-one in the local Git tree. Before refactoring for Raspberry Pi, the working Jetson code should be imported or preserved in Git as a baseline.

The live process model is simple:

```text
systemd user service
  -> /home/vlad/MILO/start_milo.sh
      -> micro_ros_agent, when CP2104 UART is present
      -> ros2 launch Orbbec DaBai camera
      -> llama-server with Gemma multimodal model
      -> python -m milo.preflight checks
      -> python -m milo.main
```

The core robot behavior is a single Python process. ROS is used as a hardware transport for camera frames and arm messages, not as the main application framework.

## Live Jetson Evidence

Observed over SSH on Jetson host `192.168.0.177`:

```text
/home/vlad/MILO -> /home/vlad/MILO_CLEAN_RC13
/home/vlad/MILO_CURRENT -> /home/vlad/MILO_CLEAN_RC13
/home/vlad/MILO_WORKING -> /home/vlad/MILO_CLEAN_RC13
```

Live relevant processes:

```text
/home/vlad/Jetson_Friend/src/control_server.py
bash /home/vlad/MILO/start_milo.sh
ros2 launch .../orbbec_camera/launch/dabai_dcw2.launch.py
/opt/ros/jazzy/lib/rclcpp_components/component_container ... camera_container
ros2 daemon --ros-domain-id 30
/home/vlad/Jetson_Friend/deps/llama.cpp/build/bin/llama-server ...
/home/vlad/Jetson_Friend/.venv/bin/python -m milo.main
```

The runtime imports models, virtualenv, Orbbec workspace, and llama.cpp from `/home/vlad/Jetson_Friend`, but the application code being executed is under `/home/vlad/MILO_CLEAN_RC13/milo`.

## Startup Chain

`milo.service`

```text
WorkingDirectory=/home/vlad/MILO
ExecStart=/home/vlad/MILO/start_milo.sh
ExecStop=/home/vlad/MILO/stop_milo.sh
Restart=on-failure
```

`start_milo.sh`

```text
load config.env
export PYTHONPATH=/home/vlad/MILO
source ROS 2 Jazzy
source interfaces workspace
source micro-ROS workspace
source Orbbec workspace
lock data/milo.lock
start or reuse micro_ros_agent over CP2104 UART
start or recover DaBai camera ROS launch
wait for color/depth topics
start or recover llama-server
run LLM/VLM/audio/vision preflight
print arm feedback
exec python -m milo.main
```

Important startup dependency: `APP_PYTHON=/home/vlad/Jetson_Friend/.venv/bin/python`, so the live `milo` package executes inside the Jetson_Friend virtualenv.

## Runtime Execution Flow

`milo.main`

```text
Config.load()
FaceUI()
RosRuntime()
VisionEngine()
LocalModel()
ObjectMemory()
VoiceIO()
FaceFollower()

Milo.start()
  FaceUI.start()
  RosRuntime.start()
  startup_lift()
  wait for camera color frame
  start vision thread
  start proactive thread
  enter voice loop
```

Main loops:

```text
voice loop:
  VoiceIO.listen()
  visual_intent()
  LocalModel.chat() or LocalModel.describe()
  VoiceIO.speak()

vision loop:
  read RosRuntime.latest_color
  VisionEngine.detect_face()
  FaceUI.set_gaze()
  FaceFollower.plan()
  RosRuntime.command_joint()

proactive loop:
  inspect fresh camera frame
  ask LocalModel.observe_person()
  speak only when proactive conditions are met
```

## Module Map

Live package: `/home/vlad/MILO_CLEAN_RC13/milo`.

| Module | Runtime role | Hardware / service dependency |
| --- | --- | --- |
| `main.py` | Orchestrates MILO, starts UI/ROS/vision/proactive/voice loops | All major services |
| `config.py` | Reads environment into immutable config | `config.env` |
| `ros_runtime.py` | Owns rclpy node, subscribes camera/arm feedback, publishes arm commands | ROS 2 Jazzy, `sensor_msgs`, `arm_msgs`, DaBai, STM32 |
| `safety.py` | Feedback-gated arm safety and local motion limits | Arm feedback messages |
| `startup.py` | Small helper for startup joint stepping | Arm startup lift |
| `tracking.py` | Converts face error into bounded joint commands | Face observation, arm safety state |
| `vision.py` | Face detection through TensorRT YuNet, ONNX DNN fallback, Haar fallback; JPEG encoding | TensorRT, CUDA, OpenCV |
| `audio.py` | Microphone VAD, Whisper STT, Piper TTS, playback device selection | sounddevice/scipy, ALSA/PipeWire, whisper.cpp, piper |
| `llm.py` | OpenAI-compatible HTTP client for local text/VLM llama-server | llama-server `/v1/chat/completions` |
| `memory.py` | Small JSON object-location memory | Local filesystem |
| `intents.py` | Detects visual/object questions | Pure Python |
| `gaze.py` | Maps camera face center into cat eye gaze | Pure Python |
| `ui.py` | Bridges runtime events to cat face UI | `cat_face.py` |
| `cat_face.py` | Display UI for animated face | Local display `:0` |
| `preflight.py` | Validates audio, model, VLM, real camera frame, detector | All selected subsystems |

Supporting files:

| File | Role |
| --- | --- |
| `scripts/serve_model.sh` | Starts `llama-server` with text model and mmproj |
| `milo.service` | systemd user unit |
| `config.env` | Live machine configuration |
| `tests/test_*.py` | Unit tests around startup, config, audio, LLM, safety, tracking, ROS decode |

## Hardware And ROS Integration

ROS domain:

```text
ROS_DOMAIN_ID=30
```

Camera:

```text
Driver: Orbbec DaBai DCW2 ROS launch
Color topic: /camera/color/image_raw
Depth topic: /camera/depth/image_raw
Compressed fallback: /camera/color/image_raw/compressed
```

Arm:

```text
micro-ROS agent: serial CP2104 at 2,000,000 baud
Command topic: /arm_joint
Feedback topic: /arm6_feedback
Raw topic: /arm6_raw
Message types: arm_msgs/msg/ArmJoint, arm_msgs/msg/ArmJoints
Allowed runtime joints: 1,3,4,5
J2: startup lift only
J6: never commanded
```

Model server:

```text
llama-server host: 127.0.0.1
llama-server port: 8081
Text/VLM model: Gemma multimodal GGUF + mmproj
```

Audio:

```text
Input hint: UM02
Output hint: UACDemo
STT: whisper.cpp
TTS: piper
```

## Dependency Graph

```text
start_milo.sh
  -> config.env
  -> ROS 2 Jazzy setup
  -> interfaces workspace
  -> micro-ROS workspace
  -> Orbbec workspace
  -> micro_ros_agent
  -> Orbbec DaBai camera launch
  -> scripts/serve_model.sh
      -> llama-server
          -> Gemma GGUF
          -> mmproj GGUF
  -> python -m milo.preflight
      -> Config
      -> VoiceIO
      -> LocalModel
      -> VisionEngine
      -> RosRuntime
  -> python -m milo.main
      -> Config
      -> FaceUI
          -> cat_face.Face
      -> RosRuntime
          -> ArmSafetyGate
          -> rclpy MultiThreadedExecutor
          -> sensor_msgs Image/CompressedImage
          -> arm_msgs ArmJoint/ArmJoints
      -> VisionEngine
          -> TensorRT/CUDA YuNet
          -> OpenCV DNN fallback
          -> Haar fallback
      -> LocalModel
          -> llama-server HTTP API
      -> VoiceIO
          -> sounddevice/scipy
          -> whisper.cpp
          -> piper
          -> ALSA/PipeWire
      -> FaceFollower
      -> ObjectMemory
      -> visual_intent
      -> camera_to_gaze
```

## Git Parity Check

Live Jetson working tree:

```text
/home/vlad/Jetson_Friend: Git branch main, clean
/home/vlad/MILO_CLEAN_RC13: live runtime code, package milo/
```

Local Mac Git tree:

```text
/Users/vladromenko/Desktop/Scoltech/Jetson friend
branch: distributed-jetson-pi-architecture
package layout: src/
```

Local Mac branch already had unrelated pending changes before this document was added:

```text
README.md
config.env.example
src/main.py
src/robotics/manager.py
start_milo_robot.sh
tests/test_face_search.py
docs/INTEGRATED_ARM.md
docs/MANIPULATION_FOUNDATION.md
src/robotics/*
start_manipulation_dry_run.sh
tests/test_manipulation.py
tools/manipulation_dry_run.py
```

Conclusion: the live implementation should be imported into Git before behavior refactoring. The safest target is a baseline commit or branch containing the exact `milo/` package and launcher from `/home/vlad/MILO_CLEAN_RC13`, plus checksums.

## Proposed Distributed Architecture

Keep the current Python application structure and introduce one boundary: `RobotBody`.

The Jetson remains the brain:

```text
conversation state
LLM/VLM orchestration
speech loop, unless audio hardware moves later
memory
high-level behavior
UI expression policy
object/person reasoning
```

Raspberry Pi + Hailo becomes the body:

```text
camera acquisition, if the camera physically moves to Pi
Hailo inference for face/person/object detection
arm microcontroller bridge, if CP2104 moves to Pi
low-level feedback normalization
optional audio IO, only if microphones/speakers move to Pi
body health/status service
```

The first abstraction should preserve call sites:

```python
body.latest_color
body.camera_fresh()
body.latest_feedback
body.ready_joints()
body.command_joint(joint, target)
body.detect_face(frame)  # later optional remote call
```

Then provide two implementations:

```text
LocalJetsonBody
  wraps current RosRuntime + VisionEngine + FaceFollower/Safety behavior

RemotePiBody
  talks to Raspberry Pi over a narrow network protocol
  keeps the same MILO main loop API
```

## Minimal-Change Migration Plan

1. Preserve the live baseline.

   Import `/home/vlad/MILO_CLEAN_RC13` into Git or archive it as a tracked baseline. Do not rewrite it during import. Verify checksums and keep `milo.service`, `start_milo.sh`, `config.env.example`, `milo/`, `scripts/`, and tests.

2. Introduce a body interface without moving hardware.

   Create `RobotBody` around the current `RosRuntime`, `VisionEngine`, and arm safety calls. Keep the implementation local on Jetson. `milo.main` should call `body`, but behavior should remain unchanged.

3. Split perception transport from perception inference.

   Keep camera frame subscription separate from face detection. This allows either:

   ```text
   Jetson subscribes ROS camera -> Jetson detects face
   Jetson receives Pi detection results -> Jetson commands behavior
   Pi publishes camera/detections -> Jetson consumes body state
   ```

4. Bring Raspberry Pi online as a passive body node.

   On the Pi, install only the runtime needed to report health and optionally run Hailo inference. Start with a read-only service:

   ```text
   /health
   /camera/status
   /detections/latest
   /arm/status
   ```

   No arm commands in the first Pi service.

5. Move Hailo inference first.

   Send camera frames to Hailo locally on the Pi only if the camera is attached there. If the camera stays on Jetson, do not stream raw video to Pi as the first version; keep Jetson vision local until the physical camera placement changes.

6. Move arm bridge only after telemetry is stable.

   If CP2104 moves to Pi, Pi should own `micro_ros_agent` and expose arm status/command endpoints. Jetson should still perform high-level safety checks before sending commands, and Pi should enforce local command limits as a second safety layer.

7. Keep ROS where it already works.

   Avoid rewriting the whole system into ROS nodes. Use ROS for hardware transports and use a small HTTP/gRPC/WebSocket or ROS topic bridge only at the body boundary.

## Recommended Network Boundary

For minimum code churn:

```text
Jetson -> Pi:
  command_joint(joint, target, runtime_ms)
  set_body_mode(passive|supervised|active)
  request_snapshot()

Pi -> Jetson:
  body_status
  camera_frame metadata
  detections
  arm feedback
  command acknowledgements
```

Preferred first protocol: HTTP + Server-Sent Events or WebSocket.

Reason: the current app is a regular Python application, not a ROS graph. A simple body service is easier to test from the Jetson and Mac. ROS can remain underneath on whichever machine owns the hardware.

## Safety Rules For The Split

1. Never command the arm from both Jetson and Pi at the same time.
2. Keep one owner for CP2104/micro-ROS agent.
3. Preserve feedback-gated commands.
4. Keep J6 disabled.
5. Keep startup lift supervised and measurable.
6. Treat stale camera/feedback as body degraded, not as a reason to guess.
7. Pi-side service must reject commands outside local limits even if Jetson sends them.

## Raspberry Pi Status

Configured SSH profile:

```text
Host rpi
HostName 192.168.1.7
User vlados
IdentityFile ~/.ssh/id_ed25519
```

Current result on 2026-09-21: SSH connection to `192.168.1.7:22` timed out. Pi setup is blocked until the reachable IP/network path is known or the Pi is brought onto the same network.

Once reachable, first setup steps should be:

```text
hostname / OS / architecture inventory
Hailo device visibility
Python version and venv plan
camera device inventory
ROS 2 availability, only if needed locally
systemd user service capability
network latency to Jetson
```

## Next Concrete Actions

1. Import or mirror `/home/vlad/MILO_CLEAN_RC13` into Git as the working baseline.
2. Add a `RobotBody` interface around the existing local runtime.
3. Add a passive Pi body service skeleton and health check only after SSH is reachable.
4. Add tests that prove the Jetson behavior can run with `LocalJetsonBody` unchanged.
5. Only then move Hailo detection or arm ownership to Pi.
