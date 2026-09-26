# Jetson-Only Architecture

## Responsibility Boundary

RC13 is deliberately a single-host system. Jetson owns model inference, audio,
camera subscriptions, display, ROS 2, memory, and arm commands. The STM32 remains
the low-level servo controller and publishes measured joint feedback through
micro-ROS.

## Startup Graph

```text
./start_milo.sh
  -> load config.env and acquire data/milo.lock
  -> apply display rotation
  -> source ROS 2 Jazzy and ros_ws/install
  -> start micro_ros_agent on CP2104 at 2,000,000 baud
  -> start Orbbec DaBai ROS launch
  -> wait for color and depth topics
  -> start llama-server with Gemma and mmproj
  -> preflight text, image, audio, camera, detector, and arm feedback
  -> python -m milo.main
```

The launcher owns every process it creates in a separate process group. `Ctrl+C`,
TERM, or launcher failure tears down those children. A file lock and process
check prevent concurrent MILO instances.

## Runtime Threads

`milo.main.Milo` coordinates four long-lived paths:

1. The ROS executor receives camera images and arm feedback and publishes bounded
   `ArmJoint` commands.
2. The vision loop consumes the latest frame, detects a face, updates screen gaze,
   and asks `FaceFollower` for a J1/J3 tracking step.
3. The voice loop records bounded speech, transcribes with Whisper, routes visual
   or ordinary intent, calls Gemma, and synthesizes the answer with Piper.
4. The proactive loop uses a fresh frame and the VLM at long intervals, then asks
   the text model to generate an appropriate short observation.

## Voice Request Flow

```text
UM02 microphone
  -> VoiceIO energy/VAD segmentation
  -> bounded temporary WAV
  -> whisper-cli
  -> visual_intent()
       -> text dialogue: LocalModel.chat()
       -> current scene: JPEG -> LocalModel.describe()
       -> object query: VLM result + ObjectMemory
  -> Piper
  -> UACDemo speaker
```

The llama.cpp server binds to `127.0.0.1`. Model output is text only and never
becomes a direct servo command.

## Motion Flow

```text
latest face observation
  -> FaceFollower.plan()
  -> ArmSafetyGate
  -> RosRuntime.command_joint()
  -> /arm_joint
  -> micro_ros_agent
  -> STM32 controller
  -> /arm6_feedback and /arm6_raw
```

Commands are absolute and bounded by joint-specific limits, step size, speed,
freshness, and measured feedback. Startup may move J2, J3, and J4 through its
separate bounded path. Autonomous tracking uses J1 and J3. J6 is never commanded.

## Module Map

| Module | Responsibility |
| --- | --- |
| `main.py` | lifecycle and voice/vision/proactive orchestration |
| `config.py` | environment parsing and invariant validation |
| `ros_runtime.py` | ROS node, camera subscriptions, arm feedback and publisher |
| `safety.py` | joint limits, feedback stability, command gating |
| `startup.py` | bounded startup stepping |
| `tracking.py` | face error to J1/J3 intent |
| `vision.py` | YuNet detection and JPEG encoding |
| `audio.py` | microphone segmentation, Whisper, Piper, audio selection |
| `llm.py` | loopback llama.cpp text/VLM client |
| `memory.py` | last-seen object locations in local JSON |
| `intents.py` | deterministic visual-intent routing |
| `gaze.py` | camera coordinates to animated-eye coordinates |
| `ui.py`, `cat_face.py` | pygame display |
| `preflight.py` | model, audio, and real-camera checks |

## Failure Model

- Missing CP2104 disables arm motion while voice and vision continue.
- Missing or stale camera prevents tracking and visual answers.
- Missing TensorRT uses ONNX/Haar face detection instead.
- A model-server failure stops startup instead of silently giving template-only
  conversation.
- Unsafe or stale arm feedback blocks that command path; no retry loop clears it.
