# Summary

Added a real DBNet-style tiny detector baseline with dataset loading, mask rasterization, pixel metrics, and a trainable segmentation loop.

# Why

The project already had cleaned detection manifests and split outputs. The next meaningful step is to enable a real detector training path instead of stopping at metadata scaffolding.

# Files Changed

- `src/dltr/models/detection/dataset.py`
- `src/dltr/models/detection/metrics.py`
- `src/dltr/models/detection/trainer.py`
- `src/dltr/models/detection/scaffold.py`
- `src/dltr/models/detection/__init__.py`
- `src/dltr/commands.py`
- `configs/detection/dbnet_baseline.yaml`
- `configs/detection/dbnet_improved.yaml`
- `tests/detection/test_dataset.py`
- `tests/detection/test_metrics.py`
- `tests/detection/test_trainer.py`
- `tests/test_detection_training_command.py`

# Verification

- `uv run pytest tests/detection/test_dataset.py tests/detection/test_metrics.py tests/detection/test_trainer.py tests/test_detection_training_command.py -q`

# Next

- Add richer DBNet components such as threshold maps and differentiable binarization branches.
- Connect detector outputs to real crop extraction for end-to-end OCR experiments.
