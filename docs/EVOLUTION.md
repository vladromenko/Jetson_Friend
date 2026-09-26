# MILO Evolution

## Stage 1: Offline Voice Prototype

[`pi_friend_hat-2`](https://github.com/vladromenko/pi_friend_hat-2) proved that a
Raspberry Pi 5 and Hailo accelerator could run an offline record, transcribe,
generate, and speak loop. It established local-first interaction but remained a
sequential voice assistant without the complete robot body.

## Stage 2: Jetson Friend

This repository added the physical companion in one machine: RGB-D perception,
multimodal Gemma, display, object memory, proactive behavior, ROS 2, startup
posture, and feedback-gated face tracking. Keeping every subsystem on Jetson made
integration direct and created the first complete MILO demonstration.

The single-host design also exposed the next engineering problem: model inference,
camera drivers, audio, display, ROS, and USB ownership shared one compute and
failure domain. Recovery and portable presentation setup were harder to isolate.

## Stage 3: Distributed MILO

[`MILO`](https://github.com/vladromenko/MILO) retained the companion behavior but
split responsibility by workload:

| Jetson brain | Raspberry Pi + Hailo body |
| --- | --- |
| Gemma dialogue, CUDA Whisper, memory, policy | camera, Hailo objects, faces |
| lifecycle, phone panel, motion decisions | microphone, Piper, speaker, display |
| ROS command publisher | exclusive CP2104 and micro-ROS ownership |

The final stage added the private presentation network, phone start/stop and
manual control, multilingual operation, person-scoped memory, expression cues,
failure isolation, and measured performance work. Median LLM completion in that
generation improved from 4.858 s to 1.814 s, and warm short-phrase STT from about
2.27 s on CPU to 0.24 s on CUDA. Those measurements belong to the distributed
repository; no equivalent historical benchmark set was retained for RC13, so
this repository does not invent one.

Stage 2 remains valuable as a simpler one-computer fallback and as the clearest
record of when MILO became an embodied robot.
