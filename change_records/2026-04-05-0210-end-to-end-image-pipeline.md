# Summary

Added a first image-mode end-to-end OCR pipeline that runs detection, crop extraction, recognition, semantic classification, and English result reporting.

# Why

The project already had separate data preparation and training baselines, but lacked a usable runtime chain from image input to structured OCR output. This change introduces that first bridge.

# Files Changed

- `src/dltr/semantic/classification.py`
- `src/dltr/semantic/__init__.py`
- `src/dltr/models/detection/inference.py`
- `src/dltr/models/recognition/inference.py`
- `src/dltr/pipeline/__init__.py`
- `src/dltr/pipeline/end_to_end.py`
- `src/dltr/commands.py`
- `src/dltr/cli.py`
- `tests/semantic/test_classification.py`
- `tests/pipeline/test_end_to_end.py`
- `tests/test_end_to_end_command.py`

# Verification

- `uv run pytest tests/semantic/test_classification.py tests/pipeline/test_end_to_end.py tests/test_end_to_end_command.py -q`

# Next

- Replace GT-derived recognition crop prep with detector-prediction-driven crop generation in the main workflow.
- Add richer end-to-end preview overlays and failure galleries.
