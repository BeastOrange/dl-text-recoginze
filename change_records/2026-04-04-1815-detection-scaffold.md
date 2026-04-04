# Summary

Implemented a DBNet-oriented detection experiment scaffold with executable preparation and reporting flows.

# Why

The thesis workflow needs an honest and runnable detection foundation before full model training is wired in.
This scaffold prepares run folders, validates required paths, writes metadata/evaluation reports, and generates export plans.

# Files Changed

- `src/dltr/models/detection/__init__.py`
- `src/dltr/models/detection/scaffold.py`
- `configs/detection/dbnet_baseline.yaml`
- `configs/detection/dbnet_improved.yaml`
- `tests/detection/test_scaffold.py`

# Verification

- `uv run pytest tests/detection/test_scaffold.py -q` -> `6 passed`

# Next

Main thread can wire these public functions into CLI commands:
- `train detector`
- `evaluate detector`
- `export onnx` (detection checkpoint path validation + plan output)
