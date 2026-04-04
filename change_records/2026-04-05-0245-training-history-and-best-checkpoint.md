# Summary

Added per-epoch training history and best-checkpoint management for both the recognition and detection training baselines.

# Why

The project already had real trainable baselines, but it still lacked the experiment traces needed for comparison, ablation, and thesis reporting. This change makes each training run produce reusable history and best-model artifacts.

# Files Changed

- `src/dltr/models/recognition/trainer.py`
- `src/dltr/models/detection/trainer.py`
- `src/dltr/commands.py`
- `tests/recognition/test_trainer.py`
- `tests/detection/test_trainer.py`
- `tests/test_recognition_training_command.py`
- `tests/test_detection_training_command.py`

# Verification

- `uv run pytest tests/recognition/test_trainer.py tests/detection/test_trainer.py tests/test_recognition_training_command.py tests/test_detection_training_command.py -q`
- `uv run pytest -q`
- `uv run ruff check .`

# Next

- Export training history plots to `reports/train/`
- Add experiment-level summary comparison between multiple runs
