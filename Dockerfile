ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake pkg-config curl ca-certificates \
    python3 python3-pip python3-opencv python3-numpy \
    libasound2 alsa-utils pulseaudio-utils portaudio19-dev \
    ffmpeg libgl1 libglib2.0-0 libsdl2-2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir sounddevice soundfile pygame pyyaml requests

RUN mkdir -p /app/deps \
 && git clone --depth 1 https://github.com/ggerganov/llama.cpp /app/deps/llama.cpp \
 && cmake -S /app/deps/llama.cpp -B /app/deps/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /app/deps/llama.cpp/build --config Release -j"$(nproc)" \
 && git clone --depth 1 https://github.com/ggerganov/whisper.cpp /app/deps/whisper.cpp \
 && cmake -S /app/deps/whisper.cpp -B /app/deps/whisper.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /app/deps/whisper.cpp/build --config Release -j"$(nproc)"

RUN set -eux; arch="$(uname -m)"; \
    case "$arch" in aarch64) piper_asset=piper_linux_aarch64.tar.gz ;; x86_64) piper_asset=piper_linux_x86_64.tar.gz ;; *) exit 1 ;; esac; \
    curl -L --fail -o /tmp/piper.tar.gz "https://github.com/rhasspy/piper/releases/latest/download/${piper_asset}"; \
    tar -xzf /tmp/piper.tar.gz -C /opt; \
    ln -sf /opt/piper/piper /usr/local/bin/piper; \
    rm /tmp/piper.tar.gz

COPY src /app/src
COPY config.env.example /app/config.env.example

ENV PYTHONUNBUFFERED=1
CMD ["python3", "/app/src/main.py"]
