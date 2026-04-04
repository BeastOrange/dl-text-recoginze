# Summary

Added a real CRNN baseline training path for recognition, including charset loading, manifest sample loading, scoring metrics, and a trainable PyTorch loop.

# Why

The project already had cleaned recognition manifests and charset outputs. The next meaningful step is to move beyond planning and enable a real baseline recognizer training path.

# Files Changed

- `src/dltr/models/recognition/charset.py`
- `src/dltr/models/recognition/dataset.py`
- `src/dltr/models/recognition/metrics.py`
- `src/dltr/models/recognition/trainer.py`
- `src/dltr/models/recognition/config.py`
- `src/dltr/models/recognition/__init__.py`
- `src/dltr/commands.py`
- `configs/recognition/crnn_baseline.yaml`
- `configs/recognition/transocr_second_pass.yaml`
- `tests/recognition/test_charset.py`
- `tests/recognition/test_dataset.py`
- `tests/recognition/test_metrics.py`
- `tests/recognition/test_config.py`
- `tests/test_recognition_training_command.py`

# Verification

- `uv run pytest tests/recognition/test_charset.py tests/recognition/test_dataset.py tests/recognition/test_metrics.py tests/recognition/test_config.py tests/test_recognition_training_command.py -q`

# Next

- Install the training extra and run a real CRNN smoke training on the prepared manifests.
- Implement the real TransOCR training loop as the next recognition milestone.
