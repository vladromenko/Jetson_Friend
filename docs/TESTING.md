# Testing

## Hardware-Free Suite

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pip install numpy scipy
.venv/bin/python -m pytest -q -p no:cacheprovider
```

The suite covers configuration invariants, visual-intent routing, noise-text
filtering, LLM response parsing, RGB image decoding, gaze transforms, startup
stepping, face tracking, and arm command safety. GitHub Actions runs it without
ROS hardware and cannot move the arm.

## Installation Check

`./install.sh --check` adds real build and asset checks: Python imports, tests,
model files, native binaries, and the ROS overlay. It does not open devices or
start model, camera, ROS, or application processes.

## Attended Acceptance

1. Inspect the mechanical path and cable slack with arm power off.
2. Run `./diagnose.sh`; confirm all expected devices and the CP2104 by-id path.
3. Start MILO and confirm color/depth, model, audio, and application camera
   preflights complete.
4. Observe the startup posture with access to the controller power cutoff.
5. Move slowly left/right and up/down in front of the camera; verify J1/J3 tracking.
6. Speak one ordinary request and one current-scene request; verify STT, local
   model generation, Piper playback, and no cloud dependency.
7. Stop with `Ctrl+C`; confirm agent, camera, model server, and application exit.
8. Reboot and confirm `milo.service` remains disabled and MILO does not move.

Record the Git tag, controller firmware, `config.env` checksum, and result. A
software test pass is not a substitute for this physical acceptance.
