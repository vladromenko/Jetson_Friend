# Operation and Troubleshooting

## Start and Stop

```bash
cd ~/Jetson_Friend
./start_milo.sh
```

The launcher remains attached and prints readiness and runtime messages. Stop it
with `Ctrl+C`; from another terminal use `./stop_milo.sh`. The optional systemd
unit is for supervised fallback operation only and stays disabled by default.

## Expected Startup

The startup order is micro-ROS agent, DaBai camera, local Gemma server, model and
device preflight, arm feedback report, then `milo.main`. If color or depth is
missing, voice may continue but tracking stays unavailable. If the language model
fails preflight, startup stops.

The bounded startup posture requests J2=115, J3=75, and J4=0 when enabled and
fresh feedback permits it. Tracking then uses J1/J3. J6 is never commanded.

## Diagnostics

```bash
./diagnose.sh
tail -n 100 logs/llm.log
tail -n 100 logs/dabai.log
tail -n 100 logs/micro_ros_agent.log
ros2 topic list
timeout 5 ros2 topic echo /arm6_feedback --once
```

`diagnose.sh` lists files, Python dependencies, ALSA devices, display sockets,
processes, ROS topics, feedback, and model-server status. It does not publish a
joint command.

## Common Problems

**No display:** verify `DISPLAY_NAME`, the X11 socket, connector name, and
`MILO_DISPLAY_ROTATION`. Run `scripts/set_display.sh normal|left|right|inverted`
only from the local graphical session.

**No audio:** compare `arecord -l` and `aplay -l` with the configured name
fragments. Replugging USB can change card numbers.

**No camera:** check the DaBai log and both ROS topics. Topic presence alone is
not enough; application preflight also waits for a frame in the MILO Python
runtime.

**Arm blocked:** check controller power, recovery-switch state, stable serial
by-id path, exclusive micro-ROS ownership, and fresh `/arm6_feedback`. Do not
repeatedly clear a fault or launch two versions of MILO.

**Model unavailable:** inspect `logs/llm.log`, confirm loopback port 8081, verify
model hashes by rerunning `scripts/bootstrap_assets.sh`, and check available RAM.

## Switching Between Stage 2 and Stage 3

Stop the active runtime completely. Stage 2 expects camera, audio, display, and
CP2104 on Jetson. Stage 3 expects those peripherals on Raspberry Pi. Move cables
only while the corresponding services and arm power are off, then follow the
target repository's hardware preflight. The two repositories use separate
directories and must never run concurrently.
