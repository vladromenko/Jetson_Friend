import os
import subprocess
import threading
from pathlib import Path


class ModelManager:
    """
    Local GGUF model manager for MILO.

    It discovers local models, identifies basic capabilities,
    finds multimodal projectors, and switches llama-server models
    without restarting the rest of MILO.
    """

    def __init__(self, ai):
        self.ai = ai
        self.lock = threading.RLock()

        self.llm_dir = Path(
            os.getenv(
                "LLM_MODEL_DIR",
                "/home/vlad/Jetson_Friend/models/llm",
            )
        )

        self.vlm_dir = Path(
            os.getenv(
                "VLM_MODEL_DIR",
                "/home/vlad/Jetson_Friend/models/vlm",
            )
        )

        self.auto_capabilities = self._env_bool(
            "MODEL_AUTO_CAPABILITIES",
            True,
        )

        self.auto_restart = self._env_bool(
            "MODEL_AUTO_RESTART",
            True,
        )

        self.auto_vision = self._env_bool(
            "MODEL_AUTO_VISION",
            True,
        )

        self.require_chat = self._env_bool(
            "MODEL_REQUIRE_CHAT",
            True,
        )

    @staticmethod
    def _env_bool(name, default=False):
        value = os.getenv(
            name,
            "1" if default else "0",
        )

        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _provider_from_name(name):
        name = str(name).lower()

        providers = (
            ("qwen", "Alibaba Qwen"),
            ("gemma", "Google"),
            ("llama", "Meta"),
            ("phi", "Microsoft"),
            ("mistral", "Mistral AI"),
            ("ministral", "Mistral AI"),
            ("deepseek", "DeepSeek"),
            ("smollm", "Hugging Face"),
        )

        for token, provider in providers:
            if token in name:
                return provider

        return "Unknown"

    @staticmethod
    def _quantization_from_name(name):
        name = str(name).upper()

        quantizations = (
            "IQ1_S",
            "IQ1_M",
            "IQ2_XXS",
            "IQ2_XS",
            "IQ2_S",
            "IQ2_M",
            "Q2_K",
            "Q3_K_S",
            "Q3_K_M",
            "Q3_K_L",
            "Q4_0",
            "Q4_1",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_0",
            "Q5_1",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
            "F16",
            "BF16",
        )

        for quantization in quantizations:
            if quantization in name:
                return quantization

        return "unknown"

    @staticmethod
    def _is_projector(path):
        name = path.name.lower()

        return (
            "mmproj" in name
            or "projector" in name
        )

    @staticmethod
    def _is_non_chat_model(path):
        name = path.name.lower()

        tokens = (
            "embedding",
            "embed",
            "reranker",
            "rerank",
            "mmproj",
            "projector",
            "clip",
        )

        return any(
            token in name
            for token in tokens
        )

    def _find_projector(self, model_path):
        model_path = Path(model_path)

        same_folder = sorted(
            path
            for path in model_path.parent.glob("*.gguf")
            if self._is_projector(path)
        )

        if same_folder:
            return same_folder[0]

        if not self.vlm_dir.exists():
            return None

        projectors = sorted(
            path
            for path in self.vlm_dir.rglob("*.gguf")
            if self._is_projector(path)
        )

        if not projectors:
            return None

        model_tokens = [
            token
            for token in model_path.stem
            .lower()
            .replace("_", "-")
            .split("-")
            if len(token) >= 3
        ]

        best = None
        best_score = 0

        for projector in projectors:
            projector_name = (
                projector.stem
                .lower()
                .replace("_", "-")
            )

            score = sum(
                1
                for token in model_tokens
                if token in projector_name
            )

            if score > best_score:
                best = projector
                best_score = score

        if best_score > 0:
            return best

        return None

    def describe_model(self, model_path):
        path = (
            Path(model_path)
            .expanduser()
            .resolve()
        )

        projector = self._find_projector(
            path
        )

        name = path.name.lower()

        is_vision = bool(
            projector
            or "-vl" in name
            or "_vl" in name
            or "vision" in name
        )

        chat = not self._is_non_chat_model(
            path
        )

        try:
            size_gb = (
                path.stat().st_size
                / (1024 ** 3)
            )
        except OSError:
            size_gb = 0.0

        if size_gb == 0:
            recommendation = "unknown"

        elif size_gb <= 2.5:
            recommendation = "fast"

        elif size_gb <= 4.5:
            recommendation = "recommended"

        elif size_gb <= 5.5:
            recommendation = "heavy"

        else:
            recommendation = "very heavy"

        return {
            "name": path.stem,
            "path": str(path),
            "provider": self._provider_from_name(
                path.name
            ),
            "quantization": self._quantization_from_name(
                path.name
            ),
            "size_gb": round(
                size_gb,
                2,
            ),
            "chat": chat,
            "vision": is_vision,
            "mmproj": (
                str(projector)
                if projector
                else None
            ),
            "recommendation": recommendation,
        }

    def scan_models(self):
        models = []
        seen = set()

        for root in (
            self.llm_dir,
            self.vlm_dir,
        ):
            if root.exists():
                paths = sorted(
                    root.rglob("*.gguf")
                )
            else:
                paths = []

            for path in paths:
                resolved = str(
                    path.resolve()
                )

                valid = (
                    resolved not in seen
                    and not self._is_projector(
                        path
                    )
                )

                if valid:
                    seen.add(
                        resolved
                    )

                    info = self.describe_model(
                        path
                    )

                    if (
                        not self.require_chat
                        or info["chat"]
                    ):
                        models.append(
                            info
                        )

        models.sort(
            key=lambda item: (
                item["provider"].lower(),
                item["name"].lower(),
            )
        )

        return models

    def current_model(self):
        return self.describe_model(
            self.ai.model
        )

    def stop_server(self):
        process = getattr(
            self.ai,
            "server_process",
            None,
        )

        if (
            process is not None
            and process.poll() is None
        ):
            process.terminate()

            try:
                process.wait(
                    timeout=5
                )

            except subprocess.TimeoutExpired:
                process.kill()

                try:
                    process.wait(
                        timeout=2
                    )

                except subprocess.TimeoutExpired:
                    pass

        self.ai.server_process = None

    def switch_model(
        self,
        model_path,
        mmproj=None,
    ):
        with self.lock:
            target = (
                Path(model_path)
                .expanduser()
                .resolve()
            )

            if not target.is_file():
                raise FileNotFoundError(
                    f"Model not found: {target}"
                )

            info = self.describe_model(
                target
            )

            if (
                self.require_chat
                and not info["chat"]
            ):
                raise ValueError(
                    "Selected GGUF does not "
                    "look like a chat model: "
                    + target.name
                )

            projector = None

            if mmproj:
                projector = (
                    Path(mmproj)
                    .expanduser()
                    .resolve()
                )

            elif (
                self.auto_capabilities
                and info.get("mmproj")
            ):
                projector = Path(
                    info["mmproj"]
                )

            if (
                projector is not None
                and not projector.is_file()
            ):
                raise FileNotFoundError(
                    "Vision projector not found: "
                    + str(projector)
                )

            old_model = self.ai.model

            old_mmproj = getattr(
                self.ai,
                "mmproj",
                "",
            )

            old_vision = getattr(
                self.ai,
                "enable_vision",
                False,
            )

            self.ai.model = str(
                target
            )

            self.ai.mmproj = (
                str(projector)
                if projector
                else ""
            )

            if self.auto_vision:
                self.ai.enable_vision = bool(
                    info["vision"]
                    and projector is not None
                )

            if self.auto_restart:
                try:
                    self.stop_server()

                    self.ai.ensure_server()

                except Exception:
                    self.ai.model = (
                        old_model
                    )

                    self.ai.mmproj = (
                        old_mmproj
                    )

                    self.ai.enable_vision = (
                        old_vision
                    )

                    try:
                        self.stop_server()

                        self.ai.ensure_server()

                    except Exception:
                        pass

                    raise

            result = self.describe_model(
                target
            )

            result["active_vision"] = bool(
                self.ai.enable_vision
            )

            return result