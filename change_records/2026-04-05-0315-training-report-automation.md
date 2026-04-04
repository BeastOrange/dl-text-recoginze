# Summary

Added automatic training curve rendering and multi-run experiment summary export for both detection and recognition baselines.

# Why

The project already writes `training_history.jsonl`, but thesis-oriented experimentation also requires visual curves and aggregated run comparisons that can be dropped into English reports and slides.

# Files Changed

- `src/dltr/visualization/training_reports.py`
- `src/dltr/visualization/__init__.py`
- `src/dltr/data/reporting.py`
- `src/dltr/models/recognition/trainer.py`
- `src/dltr/models/detection/trainer.py`
- `src/dltr/commands.py`
- `src/dltr/cli.py`
- `tests/visualization/test_training_reports.py`
- `tests/test_report_command.py`

# Verification

- `uv run pytest tests/visualization/test_training_reports.py tests/test_report_command.py -q`
- `uv run ruff check .`

# Next

- Generate consolidated English report pages under `reports/train/`
- Add side-by-side comparison of detection and recognition best runs
