# MILO

A local companion for NVIDIA Jetson Orin Nano 8 GB. The camera updates structured state; conservative event rules decide when an interaction is useful. Personal memory belongs to UUID profiles, and the LLM is called for conversation rather than every perception update.

Run on the Jetson:

```bash
cd /home/vlad/Jetson_Friend
./start.sh
```

`config.env.example` documents the tested settings. Existing installations keep their own `config.env`; model and hardware paths must match the device. `--no-face`, `--no-vision`, `--no-mic`, `--no-identity`, and `--debug` are available. Only one main instance runs at a time.

Conversation works with an unknown person, but personal memory requires an identified profile. “My name is …” starts confirmation and enrollment; matching a display name never grants access to an existing profile. With several people visible, personal context is withheld because speaker identification is not implemented.

Local memory commands include “What do you remember about me?”, “Remember that …”, “Forget that”, “Forget everything about me”, and “Don't remember this”. Profile deletion covers the active database and runtime history; external backups and legacy photo files remain separate.

```bash
.venv/bin/python -m pytest -q
```

See the [implementation and hardware report](docs/MILO_CORE_REPORT.md) for measurements, the architecture, reproduction commands, migration behavior and remaining limitations. Face recognition thresholds still require calibration; existing unaligned face references need re-enrollment.
