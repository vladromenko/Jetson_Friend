# Hardware

## Validated Platform

| Component | Interface | Owner |
| --- | --- | --- |
| NVIDIA Jetson Orin Nano 8 GB | host computer | all application services |
| Orbbec DaBai RGB-D camera | USB, ROS 2 topics | Orbbec ROS driver |
| Yahboom M3Pro controller | CP2104 USB serial, 2 Mbaud | micro-ROS agent |
| Six-axis servo arm | STM32 servo bus | controller firmware |
| UM02 microphone | USB Audio Class | `milo.audio.VoiceIO` |
| UACDemo speaker | USB Audio Class | `milo.audio.VoiceIO` |
| Portrait display | HDMI/DisplayPort, X11 | pygame UI |

The validated software base is Ubuntu 24.04, ROS 2 Jazzy under
`/opt/ros/jazzy`, and Python 3.12 with Jetson system OpenCV/NumPy/SciPy packages.

## ROS Interfaces

- `/camera/color/image_raw`
- `/camera/depth/image_raw`
- `/arm_joint`
- `/arm6_feedback`
- `/arm6_raw`

Custom `arm_msgs` source is tracked under `ros_ws/src/arm_msgs`. The Orbbec
driver, micro-ROS agent, and micro-ROS messages are fetched at pinned revisions.

## Device Discovery

```bash
ls -l /dev/serial/by-id/
arecord -l
aplay -l
xrandr --query
lsusb
```

Copy `config.env.example` to `config.env` and adjust the stable CP2104 by-id path,
audio name fragments, display output, and orientation. Do not rely on `/dev/ttyUSB0`
or ALSA card numbers because they may change after a reboot.

## Mechanical Notes

- The arm has a separate power supply; USB is communication, not servo power.
- Maintain cable slack for the full J1/J3 tracking envelope.
- The camera and screen add payload at the end of the arm.
- J6 is feedback-only in this application.
- The controller firmware is not flashed by installation or startup.
- Software checks do not replace a physical cutoff or attended commissioning.
