from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


@dataclass(frozen=True)
class Config:
    root: Path
    assistant_name: str
    ros_domain_id: int
    llm_url: str
    llm_model: Path
    vlm_mmproj: Path
    llm_max_tokens: int
    llm_timeout: int
    whisper_bin: Path
    whisper_model: Path
    whisper_language: str
    mic_sample_rate: int
    mic_block_size: int
    mic_max_seconds: float
    vad_speech_threshold: float
    vad_silence_threshold: float
    vad_end_silence: float
    vad_min_speech: float
    vad_pre_roll: float
    whisper_min_audio_rms: float
    piper_bin: Path
    piper_voice: Path
    audio_input_hint: str
    audio_output_hint: str
    face_engine: Path
    face_model: Path
    object_memory_path: Path
    face_score: float
    color_topic: str
    depth_topic: str
    camera_max_age: float
    face_smooth_alpha: float
    face_confirm_frames: int
    face_max_width: float
    face_max_height: float
    feedback_topic: str
    raw_topic: str
    command_topic: str
    allowed_joints: tuple[int, ...]
    runtime_ms: int
    max_step_deg: int
    max_speed_deg_s: float
    feedback_max_age: float
    feedback_stable_samples: int
    feedback_stability_deg: float
    feedback_max_jump_deg: float
    feedback_soft_limit_deg: int
    startup_lift_enabled: bool
    startup_j2_target: int
    startup_j3_target: int
    startup_j4_target: int
    startup_step_deg: int
    startup_runtime_ms: int
    startup_j4_runtime_ms: int
    startup_max_speed_deg_s: float
    startup_timeout_sec: float
    hard_limits: dict[int, tuple[int, int]]
    local_spans: dict[int, int]
    target_x: float
    target_y: float
    deadband_x: float
    deadband_y: float
    face_gain_x: float
    face_gain_y: float
    pan_joint: int
    pan_sign: int
    tilt_joint_a: int
    tilt_sign_a: int
    tilt_joint_b: int
    tilt_sign_b: int
    roll_joint: int
    roll_sign: int
    roll_deadband_deg: float
    face_interval: float
    cat_gaze_x_sign: int
    cat_gaze_y_sign: int
    show_face: bool

    @classmethod
    def load(cls, root: Path) -> "Config":
        allowed = tuple(int(x) for x in os.getenv("ARM_ALLOWED_JOINTS", "1,3,4,5").split(",") if x.strip())
        if set(allowed) != {1, 3, 4, 5}:
            raise ValueError("ARM_ALLOWED_JOINTS must be exactly 1,3,4,5")
        host = os.getenv("LLM_HOST", "127.0.0.1")
        port = _int("LLM_PORT", 8081)
        return cls(
            root=root,
            assistant_name=os.getenv("ASSISTANT_NAME", "MILO"),
            ros_domain_id=_int("ROS_DOMAIN_ID", 30),
            llm_url=f"http://{host}:{port}/v1",
            llm_model=_path("LLM_MODEL", "/home/vlad/Jetson_Friend/models/vlm/gemma4/gemma-4-E2B-it-Q4_0.gguf"),
            vlm_mmproj=_path("VLM_MMPROJ", "/home/vlad/Jetson_Friend/models/vlm/gemma4/mmproj-gemma-4-E2B-it-Q8_0.gguf"),
            llm_max_tokens=_int("LLM_MAX_TOKENS", 180),
            llm_timeout=_int("LLM_TIMEOUT_SEC", 120),
            whisper_bin=_path("WHISPER_BIN", "/home/vlad/Jetson_Friend/deps/whisper.cpp/build/bin/whisper-cli"),
            whisper_model=_path("WHISPER_MODEL", "/home/vlad/Jetson_Friend/models/whisper/ggml-base.en.bin"),
            whisper_language=os.getenv("WHISPER_LANGUAGE", "en"),
            mic_sample_rate=_int("MIC_SAMPLE_RATE", 44100),
            mic_block_size=_int("MIC_BLOCK_SIZE", 1024),
            mic_max_seconds=_float("MIC_MAX_SECONDS", 12.0),
            vad_speech_threshold=_float("VAD_SPEECH_THRESHOLD", 0.0020),
            vad_silence_threshold=_float("VAD_SILENCE_THRESHOLD", 0.0012),
            vad_end_silence=_float("VAD_END_SILENCE_SEC", 0.55),
            vad_min_speech=_float("VAD_MIN_SPEECH_SEC", 0.20),
            vad_pre_roll=_float("VAD_PRE_ROLL_SEC", 0.20),
            whisper_min_audio_rms=_float("WHISPER_MIN_AUDIO_RMS", 0.0008),
            piper_bin=_path("PIPER_BIN", "/home/vlad/Jetson_Friend/.venv/bin/piper"),
            piper_voice=_path("PIPER_VOICE", "/home/vlad/Jetson_Friend/models/tts/en_US-ryan-low.onnx"),
            audio_input_hint=os.getenv("AUDIO_INPUT_HINT", "C920"),
            audio_output_hint=os.getenv("AUDIO_OUTPUT_HINT", "UACDemo"),
            face_engine=_path("FACE_ENGINE", "/home/vlad/Jetson_Friend/models/vision/face_detection_yunet.engine"),
            face_model=_path("FACE_MODEL", "/home/vlad/Jetson_Friend/models/vision/face_detection_yunet.onnx"),
            object_memory_path=_path("OBJECT_MEMORY_PATH", str(Path.home() / ".local/share/milo/object_memory.json")),
            face_score=_float("FACE_SCORE", 0.55),
            color_topic=os.getenv("CAMERA_COLOR_TOPIC", "/camera/color/image_raw"),
            depth_topic=os.getenv("CAMERA_DEPTH_TOPIC", "/camera/depth/image_raw"),
            camera_max_age=_float("CAMERA_MAX_AGE_SEC", 1.0),
            face_smooth_alpha=_float("FACE_SMOOTH_ALPHA", 0.35),
            face_confirm_frames=_int("FACE_CONFIRM_FRAMES", 2),
            face_max_width=_float("FACE_MAX_WIDTH", 0.70),
            face_max_height=_float("FACE_MAX_HEIGHT", 0.85),
            feedback_topic=os.getenv("ARM_FEEDBACK_TOPIC", "/arm6_feedback"),
            raw_topic=os.getenv("ARM_RAW_TOPIC", "/arm6_raw"),
            command_topic=os.getenv("ARM_COMMAND_TOPIC", "/arm_joint"),
            allowed_joints=allowed,
            runtime_ms=_int("ARM_RUNTIME_MS", 700),
            max_step_deg=_int("ARM_MAX_STEP_DEG", 6),
            max_speed_deg_s=_float("ARM_MAX_SPEED_DEG_S", 9.0),
            feedback_max_age=_float("ARM_FEEDBACK_MAX_AGE_SEC", 1.0),
            feedback_stable_samples=_int("ARM_FEEDBACK_STABLE_SAMPLES", 3),
            feedback_stability_deg=_float("ARM_FEEDBACK_STABILITY_DEG", 3.0),
            feedback_max_jump_deg=_float("ARM_FEEDBACK_MAX_JUMP_DEG", 12.0),
            feedback_soft_limit_deg=_int("ARM_FEEDBACK_SOFT_LIMIT_DEG", 5),
            startup_lift_enabled=_bool("ARM_STARTUP_LIFT_ENABLED", True),
            startup_j2_target=_int("ARM_STARTUP_J2_TARGET", 115),
            startup_j3_target=_int("ARM_STARTUP_J3_TARGET", 75),
            startup_j4_target=_int("ARM_STARTUP_J4_TARGET", 0),
            startup_step_deg=_int("ARM_STARTUP_STEP_DEG", 4),
            startup_runtime_ms=_int("ARM_STARTUP_RUNTIME_MS", 900),
            startup_j4_runtime_ms=_int("ARM_STARTUP_J4_RUNTIME_MS", 20000),
            startup_max_speed_deg_s=_float("ARM_STARTUP_MAX_SPEED_DEG_S", 8.0),
            startup_timeout_sec=_float("ARM_STARTUP_TIMEOUT_SEC", 35.0),
            hard_limits={
                1: (_int("J1_HARD_MIN", 0), _int("J1_HARD_MAX", 180)),
                3: (_int("J3_HARD_MIN", 75), _int("J3_HARD_MAX", 135)),
                4: (_int("J4_HARD_MIN", -30), _int("J4_HARD_MAX", 115)),
                5: (_int("J5_HARD_MIN", 0), _int("J5_HARD_MAX", 270)),
            },
            local_spans={
                1: _int("J1_LOCAL_SPAN", 20),
                3: _int("J3_LOCAL_SPAN", 15),
                4: _int("J4_LOCAL_SPAN", 20),
                5: _int("J5_LOCAL_SPAN", 20),
            },
            target_x=_float("FACE_TARGET_X", 0.50),
            target_y=_float("FACE_TARGET_Y", 0.50),
            deadband_x=_float("FACE_DEADBAND_X", 0.045),
            deadband_y=_float("FACE_DEADBAND_Y", 0.045),
            face_gain_x=_float("FACE_GAIN_X", 20.0),
            face_gain_y=_float("FACE_GAIN_Y", 20.0),
            pan_joint=_int("FACE_PAN_JOINT", 1),
            pan_sign=_int("FACE_PAN_SIGN", 1),
            tilt_joint_a=_int("FACE_TILT_JOINT_A", 3),
            tilt_sign_a=_int("FACE_TILT_SIGN_A", 1),
            tilt_joint_b=_int("FACE_TILT_JOINT_B", 4),
            tilt_sign_b=_int("FACE_TILT_SIGN_B", 1),
            roll_joint=_int("FACE_ROLL_JOINT", 5),
            roll_sign=_int("FACE_ROLL_SIGN", 1),
            roll_deadband_deg=_float("FACE_ROLL_DEADBAND_DEG", 12.0),
            face_interval=_float("FACE_INTERVAL_SEC", 0.15),
            cat_gaze_x_sign=_int("CAT_GAZE_X_SIGN", -1),
            cat_gaze_y_sign=_int("CAT_GAZE_Y_SIGN", 1),
            show_face=_bool("SHOW_FACE", True),
        )
