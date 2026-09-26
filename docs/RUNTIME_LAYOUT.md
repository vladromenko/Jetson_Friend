# MILO runtime layout

There is one project directory on the Jetson:

```text
/home/vlad/Jetson_Friend
```

Everything required specifically by MILO is either tracked there or generated there by `./install.sh`.

## Tracked source

```text
milo/                         application package
ros_ws/src/arm_msgs/          custom arm ROS messages
ros_ws/dependencies.repos     pinned third-party ROS sources
scripts/                      dependency and display helpers
tests/                        unit tests
docs/                         operating and architecture notes
config.env.example            machine configuration template
requirements.txt              pinned Python-only packages
install.sh                    installation and verification
start_milo.sh                 manual launcher
stop_milo.sh                  stop helper
```

## Generated inside the same folder

```text
.venv/                        Python environment
deps/                         llama.cpp and whisper.cpp source/builds
models/                       Gemma, mmproj, Whisper, Piper and YuNet files
ros_ws/src/OrbbecSDK_ROS2/    pinned camera driver checkout
ros_ws/src/micro-ROS-Agent/   pinned micro-ROS agent checkout
ros_ws/src/micro_ros_msgs/    pinned micro-ROS messages checkout
ros_ws/build/                 ROS build output
ros_ws/install/               ROS runtime overlay
ros_ws/log/                   ROS build logs
logs/, data/, Log/            runtime output
config.env                    local machine configuration
```

These generated paths are ignored by Git because they are large, machine-specific, or reproducibly rebuildable.

## Manual-only lifecycle

Installation:

```bash
cd ~/Jetson_Friend
./install.sh
```

Static verification:

```bash
cd ~/Jetson_Friend
./install.sh --check
```

Start:

```bash
cd ~/Jetson_Friend
./start_milo.sh
```

Stop:

```bash
cd ~/Jetson_Friend
./stop_milo.sh
```

The installer places the systemd unit in the user unit directory but leaves it
disabled. The supported normal workflow is the manual start command above.

## Display

`start_milo.sh` applies the configured display orientation before opening the UI:

```text
MILO_DISPLAY_ROTATION=left
```

The helper accepts `normal`, `left`, `right`, or `inverted` and only changes the selected display output.
