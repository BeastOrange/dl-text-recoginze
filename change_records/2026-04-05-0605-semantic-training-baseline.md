# Summary

Replaced the semantic training scaffold with a real char-level semantic classifier baseline and added a runnable default semantic config.

# Why

The project already had real training baselines for detection and recognition. Semantic analysis still stopped at report generation and rule extraction, so the training path needed to be completed to match the main thesis workflow.

# Files Changed

- `src/dltr/semantic/config.py`
- `src/dltr/semantic/dataset.py`
- `src/dltr/semantic/metrics.py`
- `src/dltr/semantic/trainer.py`
- `src/dltr/semantic/__init__.py`
- `src/dltr/visualization/training_reports.py`
- `src/dltr/visualization/project_summary.py`
- `src/dltr/commands.py`
- `src/dltr/cli.py`
- `configs/semantic/char_linear_semantic.yaml`
- `tests/semantic/test_metrics.py`
- `tests/semantic/test_dataset.py`
- `tests/semantic/test_trainer.py`
- `tests/test_semantic_training_command.py`
- `tests/visualization/test_project_summary.py`
- `tests/test_project_report_command.py`
- `tests/test_report_build_all_command.py`

# Verification

- `uv run pytest tests/semantic/test_metrics.py tests/semantic/test_dataset.py tests/semantic/test_trainer.py tests/test_semantic_training_command.py -q`
- `uv run pytest -q`
- `uv run ruff check .`

# Next

- Connect semantic training outputs into project-level report automation.
- Add a stronger transformer-based semantic model later while keeping this baseline reproducible.
