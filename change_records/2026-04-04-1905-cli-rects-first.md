# Summary

Integrated the project CLI with the data, detection, recognition, semantic, demo, export, and sync scaffolds, and switched the default dataset strategy to `ReCTS-first`.

# Why

The project needs executable command-line entrypoints and a realistic default dataset policy that matches the currently accessible Chinese scene-text datasets.

# Files Changed

- `src/dltr/cli.py`
- `src/dltr/commands.py`
- `src/dltr/data/config.py`
- `src/dltr/models/detection/scaffold.py`
- `configs/data/datasets.example.yaml`
- `PLAN.md`
- `scripts/bootstrap_windows.ps1`
- `tests/test_cli.py`
- `tests/test_commands.py`
- `tests/data/test_config.py`
- `tests/data/test_validation.py`

# Verification

- `uv run pytest`
- `uv run ruff check .`
- `uv run python -m dltr demo --text "当心高压 电话13800138000"`
- `uv run python -m dltr evaluate end2end --text "营业时间09:00-21:00" --confidence 0.52 --blur-score 0.6`

# Next

- Replace scaffold-only train/eval commands with real model training loops.
- Add dataset-specific ReCTS parser and crop extraction pipeline once the downloaded files are in place.
