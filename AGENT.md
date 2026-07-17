# Agent Instructions

## Python Environment

Use the project `uv` environment for all Python commands in this repository.

- Run Python scripts with `uv run python ...` from the repository root.
- Run Python module commands with `uv run python -m ...`.
- Run tests with `uv run pytest` when pytest is available.
- Do not use system `python`, `python3`, `pip`, or global Python tools directly unless the user explicitly asks.
- Do not use `.venv/bin/python` directly unless `uv run` is not available and the user approves that fallback.

Examples:

```bash
uv run python scripts/train_openvla_ablation.py
uv run python -m py_compile models/lstm.py
uv run pytest
```
