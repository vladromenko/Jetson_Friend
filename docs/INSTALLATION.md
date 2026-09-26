# Installation

This procedure restores the Jetson-only Stage 2 runtime into one directory. It
does not start MILO, move the arm, or flash the STM32 controller.

## Prerequisites

1. NVIDIA Jetson with Ubuntu 24.04 and its working CUDA-capable software stack.
2. ROS 2 Jazzy installed at `/opt/ros/jazzy`.
3. At least 8 GB of free storage for models, source checkouts, and build output.
4. Internet access for the initial build.

## Clone the Release

```bash
git clone https://github.com/vladromenko/Jetson_Friend.git ~/Jetson_Friend
cd ~/Jetson_Friend
git checkout jetson-only-v1.0.0
```

## Review Configuration

The installer creates `config.env` from `config.env.example` once. Before the
first start, verify these machine-specific values:

```text
ARM_SERIAL
AUDIO_INPUT_HINT
AUDIO_OUTPUT_HINT
DISPLAY_NAME
MILO_DISPLAY_ROTATION
```

Keep the accepted RC13 arm targets and limits unchanged unless the mechanical
assembly is deliberately recalibrated with an attended test.

## Install

```bash
cd ~/Jetson_Friend
./install.sh
```

The script:

1. installs missing Ubuntu packages;
2. creates `.venv` with access to Jetson system packages;
3. installs pinned Python requirements;
4. fetches pinned ROS repositories and builds `ros_ws`;
5. builds pinned llama.cpp and whisper.cpp revisions;
6. downloads and SHA-256 verifies Gemma, mmproj, Whisper, Piper, and YuNet;
7. installs `milo.service` as a disabled user unit;
8. runs syntax, import, test, and asset checks;
9. exits with MILO stopped.

Model downloads are several gigabytes. A checksum mismatch is a hard failure and
must be investigated rather than bypassed.

## Verify Without Starting

```bash
./install.sh --check
systemctl --user is-enabled milo.service
systemctl --user is-active milo.service
```

Expected service state is `disabled` and `inactive`. `--check` uses
`config.env.example` when no private configuration exists and sends no hardware
commands.

## First Commissioning

Connect the camera, controller, audio devices, display, and arm power. Keep a
person at the physical cutoff. Run:

```bash
./diagnose.sh
./start_milo.sh
```

Confirm camera topics, model readiness, correct display orientation, audio, and
fresh arm feedback before entering the tracking workspace. Stop with `Ctrl+C`.
See [Testing](TESTING.md) for the complete acceptance order.

## Restore Boundary

Git restores source and manifests. It does not include model weights, compiled
dependencies, `config.env`, logs, or `data/object_memory.json`. Back up private
configuration and memory separately. Never run this Jetson-only runtime and the
distributed MILO runtime against the same controller at the same time.
