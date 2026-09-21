# MILO RC13 for Jetson

This repository is the single canonical MILO folder on the Jetson:

```text
/home/vlad/Jetson_Friend
```

The application code matches the verified RC13 package byte-for-byte. It keeps the Friday demo behavior: local Gemma text and vision, Whisper speech recognition, Piper speech, the cat display, DaBai RGB-D vision, object-location memory, emotional response handling, and feedback-gated face tracking with the robot arm.

## Important behavior

- MILO never starts as part of installation.
- The user service is disabled by default. A reboot does not start MILO.
- J2 and J6 are not autonomous tracking joints.
- Arm movement waits for measured feedback before another command is sent.
- The display rotation is applied from `MILO_DISPLAY_ROTATION` when MILO is started.
- The STM32 firmware is not flashed by these scripts.

## Install from a fresh clone

Prerequisite: Jetson Ubuntu with ROS 2 Jazzy installed at `/opt/ros/jazzy`.

```bash
git clone https://github.com/vladromenko/Jetson_Friend.git ~/Jetson_Friend
cd ~/Jetson_Friend
git checkout clean-single-folder-runtime
./install.sh
```

The installer creates the Python environment, downloads/builds the pinned ROS camera and arm dependencies in `ros_ws`, builds local model executables, downloads model files, and runs the test suite. It finishes with MILO stopped.

## Start and stop manually

Start:

```bash
cd ~/Jetson_Friend
./start_milo.sh
```

Stop from another terminal:

```bash
cd ~/Jetson_Friend
./stop_milo.sh
```

The start command stays attached to the terminal and shows the live log. `Ctrl+C` also stops every child process started by the launcher.

## Verify without starting MILO

```bash
cd ~/Jetson_Friend
./install.sh --check
```

This checks Python imports, all unit tests, model files, executables, and the generated ROS workspace. It does not open the camera, move the arm, start the model server, or launch MILO.

## Repository layout

```text
milo/                 RC13 Python runtime
ros_ws/               one camera + micro-ROS + arm workspace
scripts/              reproducible dependency and display helpers
models/               downloaded local models (generated, ignored by Git)
deps/                 llama.cpp and whisper.cpp builds (generated, ignored)
.venv/                Python environment (generated, ignored)
tests/                behavior and safety tests
docs/                 architecture, restore and migration documents
start_milo.sh         the only normal start command
stop_milo.sh          clean stop command
install.sh            install or read-only verification
config.env            local machine settings (generated, ignored)
```

See `docs/RUNTIME_LAYOUT.md` for operational details and `docs/DISTRIBUTED_ARCHITECTURE_PROPOSAL.md` before beginning the Raspberry Pi + Hailo split.
