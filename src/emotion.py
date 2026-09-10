import os
import threading
import time

import cv2
import numpy as np


class EmotionResult:
    def __init__(
        self,
        emotion="unknown",
        confidence=0.0,
        status="UNKNOWN",
        source="face",
    ):
        self.emotion = emotion
        self.confidence = float(confidence)
        self.status = status
        self.source = source

    def as_dict(self):
        return {
            "emotion": self.emotion,
            "confidence": round(
                self.confidence,
                3,
            ),
            "status": self.status,
            "source": self.source,
        }


class EmotionManager:
    """
    Lightweight facial-expression estimation for MILO.

    Important:
    Facial expression is treated only as a weak contextual signal.

    It must NOT be interpreted as a reliable statement about
    the person's internal emotional or medical state.

    The module:
    - uses the face crop already produced by Vision;
    - runs inference at a low frequency;
    - smooths predictions over time;
    - requires repeated evidence;
    - stores only temporary state;
    - does nothing if the emotion model is unavailable.
    """

    DEFAULT_LABELS = [
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "sad",
        "surprised",
        "neutral",
    ]

    def __init__(
        self,
        memory=None,
        model_path=None,
    ):
        self.memory = memory

        root = os.getenv(
            "JETSON_FRIEND_ROOT",
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            ),
        )

        self.model_path = (
            model_path
            or os.getenv(
                "FACE_EMOTION_MODEL",
                os.path.join(
                    root,
                    "models",
                    "vision",
                    "emotion.onnx",
                ),
            )
        )

        self.input_size = int(
            os.getenv(
                "EMOTION_INPUT_SIZE",
                "64",
            )
        )

        self.grayscale = (
            os.getenv(
                "EMOTION_GRAYSCALE",
                "1",
            ).strip().lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

        labels_env = os.getenv(
            "EMOTION_LABELS",
            "",
        ).strip()

        if labels_env:
            self.labels = [
                item.strip().lower()
                for item
                in labels_env.split(",")
                if item.strip()
            ]

        else:
            self.labels = list(
                self.DEFAULT_LABELS
            )

        self.min_face_quality = float(
            os.getenv(
                "EMOTION_MIN_FACE_QUALITY",
                "0.60",
            )
        )

        self.min_confidence = float(
            os.getenv(
                "EMOTION_MIN_CONFIDENCE",
                "0.55",
            )
        )

        self.strong_confidence = float(
            os.getenv(
                "EMOTION_STRONG_CONFIDENCE",
                "0.75",
            )
        )

        self.inference_interval = float(
            os.getenv(
                "EMOTION_INTERVAL_SEC",
                "1.25",
            )
        )

        self.state_ttl = float(
            os.getenv(
                "EMOTION_STATE_TTL_SEC",
                "600",
            )
        )

        self.history_seconds = float(
            os.getenv(
                "EMOTION_HISTORY_SEC",
                "8.0",
            )
        )

        self.required_observations = int(
            os.getenv(
                "EMOTION_REQUIRED_OBSERVATIONS",
                "3",
            )
        )

        self.net = None
        self.available = False
        self.backend = "disabled"

        self.lock = threading.RLock()

        self.history = []

        self.last_inference_at = 0.0

        self.last_result = EmotionResult(
            emotion="unknown",
            confidence=0.0,
            status="UNKNOWN",
        )

        self.last_stable_result = (
            EmotionResult(
                emotion="unknown",
                confidence=0.0,
                status="UNKNOWN",
            )
        )

        self.last_person_id = None

        self._load_model()

    def _load_model(self):
        if not os.path.isfile(
            self.model_path
        ):
            print(
                "Emotion model not installed yet: "
                + self.model_path,
                flush=True,
            )

            self.available = False
            self.backend = (
                "model unavailable"
            )
            return

        try:
            self.net = (
                cv2.dnn.readNetFromONNX(
                    self.model_path
                )
            )

            try:
                self.net.setPreferableBackend(
                    cv2.dnn.DNN_BACKEND_CUDA
                )

                self.net.setPreferableTarget(
                    cv2.dnn.DNN_TARGET_CUDA
                )

                self.backend = (
                    "OpenCV DNN CUDA"
                )

            except Exception:
                self.net.setPreferableBackend(
                    cv2.dnn.DNN_BACKEND_OPENCV
                )

                self.net.setPreferableTarget(
                    cv2.dnn.DNN_TARGET_CPU
                )

                self.backend = (
                    "OpenCV DNN CPU"
                )

            self.available = True

            print(
                "Emotion backend ready: "
                + self.backend,
                flush=True,
            )

        except Exception as exc:
            self.net = None
            self.available = False
            self.backend = "disabled"

            print(
                "Emotion model initialization "
                f"failed: {exc}",
                flush=True,
            )

    @staticmethod
    def _softmax(
        values,
    ):
        values = np.asarray(
            values,
            dtype=np.float32,
        ).reshape(-1)

        if values.size == 0:
            return values

        shifted = (
            values
            - np.max(
                values
            )
        )

        exp_values = np.exp(
            shifted
        )

        total = float(
            np.sum(
                exp_values
            )
        )

        if total <= 1e-8:
            return np.zeros_like(
                values
            )

        return (
            exp_values
            / total
        )

    def _prepare_face(
        self,
        face_crop,
    ):
        if (
            face_crop is None
            or face_crop.size == 0
        ):
            return None

        face = cv2.resize(
            face_crop,
            (
                self.input_size,
                self.input_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        if self.grayscale:
            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY,
            )

            face = cv2.equalizeHist(
                face
            )

            face = (
                face.astype(
                    np.float32
                )
                / 255.0
            )

            blob = face[
                None,
                None,
                :,
                :,
            ]

        else:
            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB,
            )

            face = (
                face.astype(
                    np.float32
                )
                / 255.0
            )

            face = np.transpose(
                face,
                (
                    2,
                    0,
                    1,
                ),
            )

            blob = face[
                None,
                :,
                :,
                :,
            ]

        return np.ascontiguousarray(
            blob,
            dtype=np.float32,
        )

    def _infer(
        self,
        face_crop,
    ):
        if (
            not self.available
            or self.net is None
        ):
            return EmotionResult(
                emotion="unknown",
                confidence=0.0,
                status="UNAVAILABLE",
            )

        blob = self._prepare_face(
            face_crop
        )

        if blob is None:
            return EmotionResult(
                emotion="unknown",
                confidence=0.0,
                status="NO_FACE",
            )

        try:
            self.net.setInput(
                blob
            )

            output = self.net.forward()

            raw = np.asarray(
                output,
                dtype=np.float32,
            ).reshape(-1)

            if raw.size == 0:
                return EmotionResult(
                    emotion="unknown",
                    confidence=0.0,
                    status="INVALID_OUTPUT",
                )

            if raw.size != len(
                self.labels
            ):
                print(
                    "Emotion model output size "
                    f"{raw.size} does not match "
                    f"{len(self.labels)} labels.",
                    flush=True,
                )

                return EmotionResult(
                    emotion="unknown",
                    confidence=0.0,
                    status="INVALID_OUTPUT",
                )

            if (
                np.min(raw) >= 0.0
                and np.max(raw) <= 1.0
                and abs(
                    float(
                        np.sum(raw)
                    )
                    - 1.0
                )
                < 0.15
            ):
                probabilities = raw

            else:
                probabilities = (
                    self._softmax(
                        raw
                    )
                )

            index = int(
                np.argmax(
                    probabilities
                )
            )

            confidence = float(
                probabilities[index]
            )

            emotion = (
                self.labels[index]
            )

            if (
                confidence
                < self.min_confidence
            ):
                return EmotionResult(
                    emotion="unknown",
                    confidence=confidence,
                    status="LOW_CONFIDENCE",
                )

            return EmotionResult(
                emotion=emotion,
                confidence=confidence,
                status="OBSERVED",
            )

        except Exception as exc:
            print(
                "Emotion inference warning: "
                f"{exc}",
                flush=True,
            )

            return EmotionResult(
                emotion="unknown",
                confidence=0.0,
                status="ERROR",
            )

    def _cleanup_history(
        self,
        now,
        person_id,
    ):
        cutoff = (
            now
            - self.history_seconds
        )

        self.history = [
            item
            for item
            in self.history
            if (
                item["time"]
                >= cutoff
                and item["person_id"]
                == person_id
            )
        ]

    def _add_observation(
        self,
        result,
        person_id,
    ):
        if (
            result.status
            != "OBSERVED"
        ):
            return

        now = time.monotonic()

        with self.lock:
            if (
                self.last_person_id
                != person_id
            ):
                self.history = []

            self.last_person_id = (
                person_id
            )

            self._cleanup_history(
                now,
                person_id,
            )

            self.history.append(
                {
                    "time": now,
                    "person_id": (
                        person_id
                    ),
                    "emotion": (
                        result.emotion
                    ),
                    "confidence": (
                        result.confidence
                    ),
                }
            )

    def _stable_emotion(
        self,
        person_id,
    ):
        if not person_id:
            return EmotionResult(
                emotion="unknown",
                confidence=0.0,
                status="UNKNOWN_PERSON",
            )

        now = time.monotonic()

        with self.lock:
            self._cleanup_history(
                now,
                person_id,
            )

            observations = list(
                self.history
            )

        if len(
            observations
        ) < self.required_observations:
            return EmotionResult(
                emotion="unknown",
                confidence=0.0,
                status="INSUFFICIENT_EVIDENCE",
            )

        grouped = {}

        for item in observations:
            emotion = item[
                "emotion"
            ]

            if emotion not in grouped:
                grouped[emotion] = {
                    "count": 0,
                    "confidence_sum": 0.0,
                }

            grouped[
                emotion
            ]["count"] += 1

            grouped[
                emotion
            ][
                "confidence_sum"
            ] += float(
                item["confidence"]
            )

        candidates = []

        for (
            emotion,
            values,
        ) in grouped.items():
            count = values[
                "count"
            ]

            average_confidence = (
                values[
                    "confidence_sum"
                ]
                / max(
                    count,
                    1,
                )
            )

            score = (
                count
                * average_confidence
            )

            candidates.append(
                (
                    score,
                    count,
                    average_confidence,
                    emotion,
                )
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        if not candidates:
            return EmotionResult(
                emotion="unknown",
                confidence=0.0,
                status="INSUFFICIENT_EVIDENCE",
            )

        (
            score,
            count,
            confidence,
            emotion,
        ) = candidates[0]

        if (
            count
            < self.required_observations
        ):
            return EmotionResult(
                emotion="unknown",
                confidence=confidence,
                status="INSUFFICIENT_EVIDENCE",
            )

        proportion = (
            count
            / max(
                len(observations),
                1,
            )
        )

        combined_confidence = (
            confidence
            * proportion
        )

        if (
            combined_confidence
            < self.min_confidence
        ):
            return EmotionResult(
                emotion="unknown",
                confidence=(
                    combined_confidence
                ),
                status="LOW_CONFIDENCE",
            )

        return EmotionResult(
            emotion=emotion,
            confidence=(
                combined_confidence
            ),
            status="STABLE",
        )

    def observe_from_vision(
        self,
        vision,
        person_id=None,
        force=False,
    ):
        """
        Read the latest good face from Vision,
        estimate facial expression and return
        a smoothed result.

        Normally call this about once per second.
        """

        if not self.available:
            return EmotionResult(
                emotion="unknown",
                confidence=0.0,
                status="UNAVAILABLE",
            )

        now = time.monotonic()

        if (
            not force
            and (
                now
                - self.last_inference_at
                < self.inference_interval
            )
        ):
            return self.last_stable_result

        crop, meta = (
            vision.get_face_crop(
                min_quality=(
                    self.min_face_quality
                )
            )
        )

        if (
            crop is None
            or meta is None
        ):
            self.last_result = (
                EmotionResult(
                    emotion="unknown",
                    confidence=0.0,
                    status="NO_GOOD_FACE",
                )
            )

            return self.last_result

        quality = float(
            meta.get(
                "quality",
                0.0,
            )
        )

        if (
            quality
            < self.min_face_quality
        ):
            self.last_result = (
                EmotionResult(
                    emotion="unknown",
                    confidence=0.0,
                    status="LOW_FACE_QUALITY",
                )
            )

            return self.last_result

        self.last_inference_at = (
            now
        )

        raw_result = self._infer(
            crop
        )

        self.last_result = (
            raw_result
        )

        self._add_observation(
            raw_result,
            person_id,
        )

        stable = (
            self._stable_emotion(
                person_id
            )
        )

        if (
            stable.status
            == "STABLE"
        ):
            self.last_stable_result = (
                stable
            )

            self._store_temporary_state(
                stable,
                person_id,
            )

        return stable

    def _store_temporary_state(
        self,
        result,
        person_id,
    ):
        if (
            self.memory is None
            or not person_id
            or result.status
            != "STABLE"
        ):
            return

        self.memory.set_temporary_state(
            "face_emotion",
            {
                "emotion": (
                    result.emotion
                ),
                "description": (
                    "Weak facial-expression "
                    "signal only"
                ),
            },
            person_id=person_id,
            confidence=min(
                0.75,
                result.confidence,
            ),
            source=(
                "facial_expression"
            ),
            ttl_seconds=(
                self.state_ttl
            ),
        )

    def set_explicit_emotion(
        self,
        person_id,
        emotion,
        text=None,
        confidence=0.95,
        ttl_seconds=3 * 3600,
    ):
        """
        Explicit user statements are much more
        reliable than facial-expression estimates.

        Example:
        "I'm nervous about tomorrow."
        """

        if (
            self.memory is None
            or not person_id
        ):
            return False

        emotion = str(
            emotion
        ).strip().lower()

        if not emotion:
            return False

        value = {
            "emotion": emotion,
            "description": (
                text
                or "Explicitly stated by user"
            ),
        }

        self.memory.set_temporary_state(
            "explicit_emotion",
            value,
            person_id=person_id,
            confidence=max(
                0.0,
                min(
                    1.0,
                    float(
                        confidence
                    ),
                ),
            ),
            source="explicit_speech",
            ttl_seconds=ttl_seconds,
        )

        return True

    def get_context(
        self,
        person_id=None,
    ):
        """
        Returns conservative emotional context
        suitable for the LLM prompt.
        """

        if (
            self.memory is None
            or not person_id
        ):
            return None

        explicit = (
            self.memory
            .get_temporary_state(
                "explicit_emotion",
                person_id=person_id,
                min_confidence=0.70,
            )
        )

        if explicit:
            return {
                "emotion": explicit.get(
                    "emotion",
                    "unknown",
                ),
                "confidence": 0.95,
                "source": (
                    "explicit_speech"
                ),
                "instruction": (
                    "The user explicitly "
                    "described this emotional "
                    "context. Respond naturally "
                    "if relevant, but do not "
                    "repeat it unnecessarily."
                ),
            }

        face_state = (
            self.memory
            .get_temporary_state(
                "face_emotion",
                person_id=person_id,
                min_confidence=0.55,
            )
        )

        if not face_state:
            return None

        emotion = face_state.get(
            "emotion",
            "unknown",
        )

        return {
            "emotion": emotion,
            "confidence": 0.55,
            "source": (
                "facial_expression"
            ),
            "instruction": (
                "This is only a weak visual "
                "cue. Never state that the "
                f"user is definitely {emotion}. "
                "Use it only to slightly adjust "
                "tone or, when appropriate, "
                "ask a gentle neutral question."
            ),
        }

    def reset_person(
        self,
        person_id=None,
    ):
        with self.lock:
            if (
                person_id is None
                or self.last_person_id
                == person_id
            ):
                self.history = []

                self.last_person_id = (
                    None
                )

                self.last_result = (
                    EmotionResult()
                )

                self.last_stable_result = (
                    EmotionResult()
                )

    def status(self):
        return {
            "available": (
                self.available
            ),
            "backend": self.backend,
            "model_path": (
                self.model_path
            ),
            "interval_seconds": (
                self.inference_interval
            ),
            "min_face_quality": (
                self.min_face_quality
            ),
            "min_confidence": (
                self.min_confidence
            ),
            "last_raw": (
                self.last_result
                .as_dict()
            ),
            "last_stable": (
                self.last_stable_result
                .as_dict()
            ),
        }