# Restore checklist

Canonical folder:

```text
/home/vlad/Jetson_Friend
```

Canonical baseline branch:

```text
clean-single-folder-runtime
```

## Restore on the same Jetson

```bash
cd ~
git clone https://github.com/vladromenko/Jetson_Friend.git Jetson_Friend
cd Jetson_Friend
git checkout clean-single-folder-runtime
./install.sh
```

The installer does not start MILO and does not enable its systemd service.

## Verify the restored installation

```bash
cd ~/Jetson_Friend
./install.sh --check
systemctl --user is-enabled milo.service
systemctl --user is-active milo.service
```

Expected service state:

```text
disabled
inactive
```

## Manual hardware test

Connect the DaBai camera, CP2104 arm controller, UM02 microphone, UACDemo speaker, and display. Then run:

```bash
cd ~/Jetson_Friend
./start_milo.sh
```

Confirm these lines appear before interacting with the robot:

```text
[READY] DaBai color
[READY] DaBai depth
[READY] local Gemma server
[READY] application receives DaBai color frames
[READY] MILO Clean RC13 runtime
```

Also confirm that the cat display is upright, speech input/output works, the face is detected, and the arm follows with J2/J6 stationary.

Stop with `Ctrl+C` or, from another terminal:

```bash
cd ~/Jetson_Friend
./stop_milo.sh
```

## Recovery facts

- Heavy model files and compiled dependencies are generated locally and intentionally excluded from Git.
- Their exact download/build procedure is stored in `scripts/bootstrap_assets.sh` and `scripts/bootstrap_ros.sh`.
- Third-party ROS repositories are pinned to exact Git commits in `ros_ws/dependencies.repos`.
- Custom `arm_msgs` definitions are stored directly in Git under `ros_ws/src/arm_msgs`.
- `config.env` is machine-local; `config.env.example` is the reproducible template.
- No restore step flashes the STM32 controller.
