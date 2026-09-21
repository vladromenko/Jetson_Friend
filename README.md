# MILO Clean RC13

MILO runs on the existing STM32 firmware. This configuration does not flash the controller.

Key behavior:
- DaBai raw/compressed color fallback remains enabled.
- Local Gemma text+vision, UM02 USB microphone, UACDemo/PipeWire speaker and cat UI are preserved.
- Face tracking uses J1 for horizontal pan and J3 for limited vertical movement. J2, J4, and J6 are not autonomous tracking joints.
- J1 pan has a fixed configurable sign. RC9 logs show that decreasing J1 from 104 to 80 degrees moved the face from x=0.54 toward x=0.90. RC13 increases J1 for a face on the right.
- J4 currently reports about -31 degrees and remains locked. The installed firmware rejects small negative-angle commands for J4.
- Eye gaze is mirrored horizontally for the user-facing cat display, with quick easing. The eyes respond to the face before the slower arm follows.
- Face coordinates reset after detection is lost. A face at the edge of the image can be recentered; only an excessively large face pauses motion.
- Tracking uses small feedback-confirmed commands; J2 and J6 remain stationary in MILO.
- Voice replies now give more specific support for sadness and share the user's good news. A generic emotional reply is retried once.
- Questions such as "Where are my glasses?" inspect the current camera image. Confirmed sightings are saved to a persistent local file and may be recalled later as last seen locations. Unseen items are not guessed.
- Motion is feedback-gated: MILO sends one absolute command, waits for measured feedback to reach the target (or timeout), then sends the next command. It no longer stacks commands while a servo is still moving.
- `/arm6_feedback` is the angle source. `/arm6_raw` is also subscribed so abnormal calibration/physical positions can be distinguished from UART corruption.
- A bad/out-of-range joint is blocked independently; other valid axes can continue.
- The current raised posture is approximately J2=115, J3=106, J4=-31, J6=27. No startup scan or startup pose commands are sent.
- J3 is limited to the raised range (100 degrees and above); J2 manual commands cannot lower it past 115 degrees, and J6 manual commands are blocked.

Install once:

```bash
cd ~/MILO_CLEAN_RC13 && ./install.sh
```

Start:

```bash
cd ~/MILO_CLEAN_RC13 && ./start_milo.sh
```
