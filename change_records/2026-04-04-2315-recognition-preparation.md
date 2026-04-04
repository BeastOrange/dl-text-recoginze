# Summary

Added a real recognition data-preparation pipeline that builds per-dataset manifests, combines them, generates a charset file, and creates train/val/test splits.

# Why

The project already has real `ReCTS` and `ShopSign` data on disk. The next step toward actual training is to convert those datasets into unified recognition inputs instead of staying at per-dataset scaffold level.

# Files Changed

- `src/dltr/data/preparation.py`
- `src/dltr/data/__init__.py`
- `src/dltr/commands.py`
- `src/dltr/cli.py`
- `configs/recognition/crnn_baseline.yaml`
- `configs/recognition/transocr_second_pass.yaml`
- `tests/data/test_preparation.py`
- `tests/test_data_pipeline_command.py`

# Verification

- `uv run pytest tests/data/test_preparation.py tests/test_data_pipeline_command.py -q`
- `uv run python scripts/run_dltr.py data prepare-recognition --config configs/data/datasets.example.yaml --datasets rects shopsign`

# Next

- Use the generated train/val/test manifests and charset in the real recognition training loop.
- Add dataset-specific crop extraction and detection-to-recognition sample generation.
