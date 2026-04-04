# Summary

Added a detection data-preparation pipeline that converts `ReCTS` and `ShopSign` into unified detection manifests and train/val/test splits.

# Why

The project already has usable Chinese scene-text datasets on disk. Detection training needs a normalized annotation representation before the real detector training loop can be implemented cleanly.

# Files Changed

- `src/dltr/data/detection_preparation.py`
- `src/dltr/data/__init__.py`
- `src/dltr/commands.py`
- `src/dltr/cli.py`
- `tests/data/test_detection_preparation.py`
- `tests/test_detection_pipeline_command.py`

# Verification

- `uv run pytest tests/data/test_detection_preparation.py tests/test_detection_pipeline_command.py -q`
- `uv run python scripts/run_dltr.py data prepare-detection --config configs/data/datasets.example.yaml --datasets rects shopsign`

# Next

- Connect detection split outputs to the actual detector training loop.
- Add richer ignore-text handling and detection quality statistics.
