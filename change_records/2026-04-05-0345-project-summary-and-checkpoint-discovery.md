# Summary

Added best-checkpoint auto-discovery for image-mode end-to-end evaluation and a project-level training summary report command.

# Why

The training/reporting workflow already produced per-run summaries, but end-to-end usage still required manual checkpoint paths and there was no combined project view across detection and recognition runs.

# Files Changed

- `src/dltr/pipeline/checkpoints.py`
- `src/dltr/pipeline/__init__.py`
- `src/dltr/visualization/project_summary.py`
- `src/dltr/visualization/__init__.py`
- `src/dltr/commands.py`
- `src/dltr/cli.py`
- `tests/pipeline/test_checkpoint_resolution.py`
- `tests/visualization/test_project_summary.py`
- `tests/test_project_report_command.py`
- `tests/test_end_to_end_command.py`

# Verification

- `uv run pytest tests/pipeline/test_checkpoint_resolution.py tests/visualization/test_project_summary.py tests/test_project_report_command.py tests/test_end_to_end_command.py -q`

# Next

- Auto-discover the latest/best run when report commands receive only task roots.
- Add a single report page linking detection, recognition, and end-to-end artifacts together.
