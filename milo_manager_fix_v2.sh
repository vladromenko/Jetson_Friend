#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${JETSON_FRIEND_ROOT:-$HOME/Jetson_Friend}"
cd "$ROOT"

log(){ printf '\n============================================================\n== %s\n============================================================\n' "$*"; }
ok(){ printf 'OK: %s\n' "$*"; }
warn(){ printf 'WARN: %s\n' "$*" >&2; }
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }

ORIGINAL="$ROOT/.milo_control_backup_20260910_183745/src/model_manager.py"

[ -f "$ORIGINAL" ] || die "Original model_manager.py backup not found: $ORIGINAL"
[ -f src/control_server.py ] || die "src/control_server.py missing"
[ -f src/model_registry.py ] || die "src/model_registry.py missing"
[ -x .venv/bin/python ] || die ".venv/bin/python missing"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/.milo_manager_fix2_backup_$STAMP"
mkdir -p "$BACKUP"
cp src/model_manager.py src/control_server.py "$BACKUP"/ 2>/dev/null || true
ok "Backup created: $BACKUP"

log "1/5 - Restore original runtime ModelManager"
cp "$ORIGINAL" "$ROOT/src/model_manager.py"
./.venv/bin/python -m py_compile src/model_manager.py
ok "Original runtime ModelManager restored"

log "2/5 - Install separate panel model manager"

cat > "$ROOT/src/panel_model_manager.py" <<'PY'
#!/usr/bin/env python3

import os
import subprocess
import time
from pathlib import Path

from model_registry import ModelRegistry


ROOT = Path(
    os.getenv(
        "JETSON_FRIEND_ROOT",
        Path(__file__).resolve().parents[1],
    )
).resolve()

CONFIG = ROOT / "config.env"


class PanelModelManager:
    def __init__(self):
        self.registry = ModelRegistry()

    def _read_config(self):
        data = {}

        if not CONFIG.exists():
            return data

        for raw in CONFIG.read_text(
            encoding="utf-8"
        ).splitlines():
            line = raw.strip()

            if (
                line
                and not line.startswith("#")
                and "=" in line
            ):
                key, value = line.split("=", 1)
                data[key] = value

        return data

    def _write_config(self, updates):
        if CONFIG.exists():
            lines = CONFIG.read_text(
                encoding="utf-8"
            ).splitlines()
        else:
            lines = []

        seen = set()
        output = []

        for raw in lines:
            stripped = raw.strip()
            key = None

            if (
                stripped
                and not stripped.startswith("#")
                and "=" in stripped
            ):
                key = stripped.split("=", 1)[0]

            if key in updates:
                if key not in seen:
                    output.append(
                        f"{key}={updates[key]}"
                    )
                    seen.add(key)
            else:
                output.append(raw)

        for key, value in updates.items():
            if key not in seen:
                output.append(
                    f"{key}={value}"
                )

        CONFIG.write_text(
            "\n".join(output).rstrip() + "\n",
            encoding="utf-8",
        )

    def current(self):
        config = self._read_config()

        return {
            "text_model": config.get(
                "ACTIVE_TEXT_MODEL",
                "",
            ),
            "text_model_path": config.get(
                "LLM_MODEL",
                "",
            ),
            "vlm_model": config.get(
                "ACTIVE_VLM_MODEL",
                "",
            ),
            "vlm_model_path": config.get(
                "VLM_MODEL",
                "",
            ),
            "vlm_mmproj": config.get(
                "VLM_MMPROJ",
                "",
            ),
        }

    def select_text(self, model_id):
        info = self.registry.get(model_id)

        if info.get("type") != "text":
            raise ValueError(
                "Not a text model"
            )

        if not info["installed"]:
            raise FileNotFoundError(
                "Model is not installed"
            )

        model_path = info["paths"][0]

        self._write_config(
            {
                "LLM_MODEL": model_path,
                "ACTIVE_TEXT_MODEL": model_id,
            }
        )

        return info

    def select_vlm(self, model_id):
        info = self.registry.get(model_id)

        if info.get("type") != "vision":
            raise ValueError(
                "Not a vision model"
            )

        if not info["installed"]:
            raise FileNotFoundError(
                "Model is not installed"
            )

        model_path = None
        mmproj = None

        for path in info["paths"]:
            if "mmproj" in path.lower():
                mmproj = path
            elif model_path is None:
                model_path = path

        if not model_path or not mmproj:
            raise ValueError(
                "VLM entry must contain model + mmproj"
            )

        self._write_config(
            {
                "VLM_MODEL": model_path,
                "VLM_MMPROJ": mmproj,
                "ACTIVE_VLM_MODEL": model_id,
                "VLM_ENABLE": "1",
            }
        )

        return info

    def milo_running(self):
        result = subprocess.run(
            [
                "pgrep",
                "-af",
                "src/main.py",
            ],
            text=True,
            capture_output=True,
        )

        return (
            result.returncode == 0
            and bool(result.stdout.strip())
        )

    def start_milo(self):
        if self.milo_running():
            return "already running"

        log_path = ROOT / "milo_runtime.log"

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as log_file:
            subprocess.Popen(
                [
                    "bash",
                    str(ROOT / "start.sh"),
                ],
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        return "started"

    def stop_milo(self):
        subprocess.run(
            [
                "pkill",
                "-f",
                "src/main.py",
            ],
            check=False,
        )

        return "stopped"

    def restart_milo(self):
        self.stop_milo()
        time.sleep(1)
        return self.start_milo()
PY

./.venv/bin/python -m py_compile src/panel_model_manager.py
ok "Separate panel manager installed"

log "3/5 - Rewire control panel to separate manager"

./.venv/bin/python - <<'PY'
from pathlib import Path

path = Path("src/control_server.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    "from model_manager import ModelManager",
    "from panel_model_manager import PanelModelManager",
)

text = text.replace(
    "manager = ModelManager()",
    "manager = PanelModelManager()",
)

path.write_text(
    text,
    encoding="utf-8",
)

print("control_server.py updated")
PY

./.venv/bin/python -m py_compile src/control_server.py src/model_registry.py
ok "Control panel rewired"

log "4/5 - Validate both managers"

./.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "src")

from model_manager import ModelManager
from panel_model_manager import PanelModelManager

class DummyAI:
    pass

runtime = ModelManager(DummyAI())
print("Runtime ModelManager(ai): OK")

panel = PanelModelManager()
print("PanelModelManager(): OK")
print("Current:", panel.current())
PY

systemctl --user daemon-reload || true
systemctl --user restart milo-control.service || true
sleep 2

if systemctl --user is-active --quiet milo-control.service; then
    ok "MILO Control service active"
else
    warn "MILO Control service is not active"
    systemctl --user --no-pager status milo-control.service || true
fi

log "5/5 - Smoke test MILO"

set +e
timeout 20 ./start.sh >/tmp/milo_fix2_smoke.log 2>&1
RC=$?
set -e

if grep -q "Traceback (most recent call last)" /tmp/milo_fix2_smoke.log; then
    cat /tmp/milo_fix2_smoke.log
    die "MILO produced a traceback"
fi

if grep -q "TypeError: ModelManager" /tmp/milo_fix2_smoke.log; then
    cat /tmp/milo_fix2_smoke.log
    die "ModelManager API conflict remains"
fi

ok "No Python traceback during MILO startup smoke test"

echo
echo "Last MILO startup lines:"
tail -50 /tmp/milo_fix2_smoke.log || true

echo
echo "============================================================"
echo "MILO manager fix v2 completed."
echo "Runtime ModelManager and web-panel manager are now separated."
echo "Backup: $BACKUP"
echo
echo "Run MILO:"
echo "  cd $ROOT && ./start.sh"
echo
echo "Control panel:"
echo "  http://$(hostname -I | awk '{print $1}'):8765"
echo "============================================================"
