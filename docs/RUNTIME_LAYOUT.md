# MILO runtime layout

The Jetson runtime has one primary folder:

```text
/home/vlad/Jetson_Friend
```

Start MILO from this folder:

```bash
cd ~/Jetson_Friend
./start_milo.sh
```

Systemd also points directly to this folder.

Tracked in Git:

```text
milo/                 Python runtime package
scripts/              launch helpers
start_milo.sh         main launcher
stop_milo.sh          stop helper
install.sh            readiness check
milo.service          user service definition
tests/                unit tests
README.md             operator notes
config.env.example    example machine configuration
```

Local-only runtime assets expected by `config.env`:

```text
.venv/                Python environment
models/               LLM, VLM, Whisper, Piper, vision models
deps/                 llama.cpp, whisper.cpp and other local builds
orbbec_ws/            Orbbec ROS workspace, if kept inside this folder
logs/, data/, Log/    generated at runtime
config.env            local machine configuration
```

Do not commit `config.env`, logs, caches, virtualenvs, models, or build outputs.
