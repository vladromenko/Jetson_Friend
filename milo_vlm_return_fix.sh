#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${JETSON_FRIEND_ROOT:-$HOME/Jetson_Friend}"
cd "$ROOT"

[ -f src/ai.py ] || { echo "ERROR: src/ai.py missing"; exit 1; }
[ -x .venv/bin/python ] || { echo "ERROR: .venv missing"; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
cp src/ai.py "src/ai.py.bak_vlm_return_$STAMP"

./.venv/bin/python - <<'PY'
from pathlib import Path

path = Path("src/ai.py")
text = path.read_text(encoding="utf-8")

start = text.find("    def ask_visual(")
end = text.find("\n    def _fallback(", start)

if start < 0 or end < 0:
    raise SystemExit("ERROR: ask_visual() block not found")

block = text[start:end]

count = block.count('"memory": None')
if count == 0:
    print("No memory=None values found; nothing changed.")
else:
    block = block.replace('"memory": None', '"memory": {}')
    text = text[:start] + block + text[end:]
    path.write_text(text, encoding="utf-8")
    print(f"Patched {count} VLM return value(s): memory None -> empty dict")

PY

./.venv/bin/python -m py_compile src/ai.py src/main.py

echo
echo "OK: VLM return fix installed."
echo "Backup: src/ai.py.bak_vlm_return_$STAMP"
echo
echo "Now run:"
echo "cd ~/Jetson_Friend && ./start.sh"
