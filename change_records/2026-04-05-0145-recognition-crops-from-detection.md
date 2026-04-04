# Summary

Added a crop-based recognition preparation pipeline that extracts text-line crops directly from detection split manifests.

# Why

The project already had separate detection and recognition baselines, but the two stages were not yet connected through a shared data flow. Crop extraction from detection manifests is the first concrete bridge between them.

# Files Changed

- `src/dltr/data/recognition_crops.py`
- `src/dltr/data/__init__.py`
- `src/dltr/commands.py`
- `src/dltr/cli.py`
- `tests/data/test_recognition_crops.py`
- `tests/test_recognition_crop_command.py`

# Verification

- `uv run pytest tests/data/test_recognition_crops.py tests/test_recognition_crop_command.py -q`
- `uv run python scripts/run_dltr.py data prepare-recognition-crops --max-samples 8`

# Next

- Replace the old full-image recognition preparation path with the crop-based path as the mainline workflow.
- Add crop quality filtering and perspective-normalization diagnostics.
