#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${JETSON_FRIEND_ROOT:-$HOME/Jetson_Friend}"
cd "$ROOT"

log(){ printf '\n============================================================\n== %s\n============================================================\n' "$*"; }
ok(){ printf 'OK: %s\n' "$*"; }
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -f src/ai.py ] || die "src/ai.py missing"
[ -f src/model_manager.py ] || die "src/model_manager.py missing"
[ -f src/model_registry.py ] || die "src/model_registry.py missing"
[ -f src/control_server.py ] || die "src/control_server.py missing"
[ -f config.env ] || die "config.env missing"
[ -x .venv/bin/python ] || die ".venv/bin/python missing"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/.milo_manager_repair_backup_$STAMP"
mkdir -p "$BACKUP"
cp src/model_manager.py src/control_server.py src/ai.py config.env "$BACKUP"/
ok "Backup created: $BACKUP"

log "1/5 - Locate original MILO ModelManager"

ORIGINAL=""
for d in "$ROOT"/.milo_control_backup_*/src/model_manager.py; do
    if [ -f "$d" ] && grep -q 'def switch_model' "$d" && grep -q 'def scan_models' "$d"; then
        ORIGINAL="$d"
    fi
done

[ -n "$ORIGINAL" ] || die "Could not find original ModelManager in .milo_control_backup_*"
echo "Using original: $ORIGINAL"

log "2/5 - Build compatibility ModelManager"

./.venv/bin/python - "$ORIGINAL" <<'PY'
import sys
from pathlib import Path

root = Path.home() / "Jetson_Friend"
original = Path(sys.argv[1])
target = root / "src" / "model_manager.py"

text = original.read_text(encoding="utf-8")

if "from model_registry import ModelRegistry" not in text:
    marker = "from pathlib import Path\n"
    text = text.replace(marker, marker + "\nfrom model_registry import ModelRegistry\n", 1)

text = text.replace(
    "    def __init__(self, ai):\n        self.ai = ai\n",
    "    def __init__(self, ai=None):\n        self.ai = ai\n        self.registry = ModelRegistry()\n",
    1,
)

text = text.replace(
    '''    def current_model(self):
        return self.describe_model(
            self.ai.model
        )
''',
    '''    def current_model(self):
        if self.ai is None:
            config = self._read_config()
            model = config.get("LLM_MODEL", "")
            if not model:
                return {}
            return self.describe_model(model)

        return self.describe_model(
            self.ai.model
        )
''',
    1,
)

text = text.replace(
    '''    def stop_server(self):
        process = getattr(
            self.ai,
            "server_process",
            None,
        )
''',
    '''    def stop_server(self):
        if self.ai is None:
            return

        process = getattr(
            self.ai,
            "server_process",
            None,
        )
''',
    1,
)

extra = '''

    def _config_path(self):
        return Path(
            os.getenv(
                "JETSON_FRIEND_CONFIG",
                str(Path(__file__).resolve().parents[1] / "config.env"),
            )
        )

    def _read_config(self):
        path = self._config_path()
        data = {}

        if not path.exists():
            return data

        for raw in path.read_text(encoding="utf-8").splitlines():
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
        path = self._config_path()

        if path.exists():
            lines = path.read_text(
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

        path.write_text(
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
            raise ValueError("Not a text model")

        if not info["installed"]:
            raise FileNotFoundError("Model is not installed")

        path = info["paths"][0]

        self._write_config(
            {
                "LLM_MODEL": path,
                "ACTIVE_TEXT_MODEL": model_id,
            }
        )

        if self.ai is not None:
            self.switch_model(path)

        return info

    def select_vlm(self, model_id):
        info = self.registry.get(model_id)

        if info.get("type") != "vision":
            raise ValueError("Not a vision model")

        if not info["installed"]:
            raise FileNotFoundError("Model is not installed")

        model_path = None
        mmproj = None

        for path in info["paths"]:
            if "mmproj" in path.lower():
                mmproj = path
            elif model_path is None:
                model_path = path

        if not model_path or not mmproj:
            raise ValueError("VLM entry must contain model + mmproj")

        self._write_config(
            {
                "VLM_MODEL": model_path,
                "VLM_MMPROJ": mmproj,
                "ACTIVE_VLM_MODEL": model_id,
                "VLM_ENABLE": "1",
            }
        )

        if self.ai is not None:
            self.ai.vlm_model = model_path
            self.ai.mmproj = mmproj

        return info

    def milo_running(self):
        result = subprocess.run(
            ["pgrep", "-af", "src/main.py"],
            text=True,
            capture_output=True,
        )

        return result.returncode == 0 and bool(result.stdout.strip())

    def start_milo(self):
        if self.milo_running():
            return "already running"

        root = Path(__file__).resolve().parents[1]
        log_path = root / "milo_runtime.log"

        with log_path.open("a", encoding="utf-8") as log_file:
            subprocess.Popen(
                ["bash", str(root / "start.sh")],
                cwd=root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        return "started"

    def stop_milo(self):
        subprocess.run(
            ["pkill", "-f", "src/main.py"],
            check=False,
        )

        return "stopped"

    def restart_milo(self):
        self.stop_milo()

        import time
        time.sleep(1)

        return self.start_milo()
'''

if "    def select_text(self, model_id):" not in text:
    text = text.rstrip() + extra + "\n"

target.write_text(text, encoding="utf-8")
print("Compatibility ModelManager written")
PY

log "3/5 - Update panel behavior"

./.venv/bin/python - <<'PY'
from pathlib import Path

path = Path("src/control_server.py")
text = path.read_text(encoding="utf-8")

if 'result["restart_required"] = manager.milo_running()' not in text:
    text = text.replace(
        'return manager.select_text(body.model_id)',
        'result = manager.select_text(body.model_id)\n        result["restart_required"] = manager.milo_running()\n        return result',
        1,
    )
    text = text.replace(
        'return manager.select_vlm(body.model_id)',
        'result = manager.select_vlm(body.model_id)\n        result["restart_required"] = manager.milo_running()\n        return result',
        1,
    )

path.write_text(text, encoding="utf-8")
print("Control panel compatibility updated")
PY

log "4/5 - Validate merged runtime"

./.venv/bin/python -m py_compile src/model_manager.py src/control_server.py src/ai.py src/main.py

./.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "src")
from model_manager import ModelManager

panel_manager = ModelManager()
print("Panel manager: OK")
print("Current:", panel_manager.current())

class DummyAI:
    pass

dummy = DummyAI()
runtime_manager = ModelManager(dummy)
print("Runtime manager constructor: OK")
PY

systemctl --user restart milo-control.service || true
sleep 1

log "5/5 - Smoke test MILO startup"

set +e
timeout 18 ./start.sh >/tmp/milo_repair_smoke.log 2>&1
RC=$?
set -e

if grep -q "TypeError: ModelManager.__init__" /tmp/milo_repair_smoke.log; then
    cat /tmp/milo_repair_smoke.log
    die "ModelManager constructor conflict still exists"
fi

if grep -q "Traceback (most recent call last)" /tmp/milo_repair_smoke.log; then
    cat /tmp/milo_repair_smoke.log
    die "MILO produced a traceback during smoke test"
fi

ok "No Python traceback during startup smoke test"

echo
echo "Last startup lines:"
tail -40 /tmp/milo_repair_smoke.log || true

echo
echo "============================================================"
echo "MILO ModelManager compatibility repair completed."
echo "Backup: $BACKUP"
echo
echo "Now run normally:"
echo "  cd $ROOT && ./start.sh"
echo
echo "Control panel:"
echo "  http://$(hostname -I | awk '{print $1}'):8765"
echo "============================================================"
