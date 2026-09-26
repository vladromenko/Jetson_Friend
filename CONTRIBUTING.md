# Contributing

Keep RC13 behavior changes separate from documentation and installation work.
Run `.venv/bin/python -m pytest -q -p no:cacheprovider` before proposing a change.
Motion changes require tests plus an attended hardware report that names the
joint, initial pose, target, firmware, and measured feedback. Never commit models,
logs, recordings, `config.env`, object memory, or build output.
