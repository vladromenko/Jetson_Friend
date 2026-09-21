# Restore checklist

The clean MILO runtime code is saved on branch `clean-single-folder-runtime`.

Canonical Jetson folder:

```text
/home/vlad/Jetson_Friend
```

Restore code:

```bash
cd ~
git clone https://github.com/vladromenko/Jetson_Friend.git Jetson_Friend
cd Jetson_Friend
git checkout clean-single-folder-runtime
cp config.env.example config.env
```

Start manually:

```bash
cd ~/Jetson_Friend
./start_milo.sh
```

Install the user service:

```bash
mkdir -p ~/.config/systemd/user
cp ~/Jetson_Friend/milo.service ~/.config/systemd/user/milo.service
systemctl --user daemon-reload
systemctl --user enable milo.service
systemctl --user start milo.service
```

Important: Git stores the MILO code, service, scripts, tests, and documentation. It does not store local heavy runtime assets. Before a full robot start, `./install.sh` currently expects these local files inside `~/Jetson_Friend`:

```text
deps/llama.cpp/build/bin/llama-server
models/vlm/gemma4/gemma-4-E2B-it-Q4_0.gguf
models/vlm/gemma4/mmproj-gemma-4-E2B-it-Q8_0.gguf
deps/whisper.cpp/build/bin/whisper-cli
models/whisper/ggml-base.en.bin
models/tts/en_US-ryan-low.onnx
models/vision/face_detection_yunet.onnx
```

ROS workspaces expected by `config.env`:

```text
/home/vlad/microros_ws
/home/vlad/milo2_interfaces_ws
/home/vlad/Jetson_Friend/orbbec_ws
```

Current validation result after cleanup:

```text
Python environment recreated in ~/Jetson_Friend/.venv
51 unit tests pass
start is blocked until the missing model/dependency assets above are restored
```
