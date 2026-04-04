# Summary

Removed the unnecessary hard dependency on `torch` from the detection metrics test so CI can run under the `dev` dependency set.

# Why

The GitHub Actions workflow installs only the `dev` extra, while `tests/detection/test_metrics.py` imported `torch` at collection time. The metric logic itself does not require `torch`, so the test should not force the training dependency set.

# Files Changed

- `src/dltr/models/detection/metrics.py`
- `tests/detection/test_metrics.py`

# Verification

- `uv run pytest tests/detection/test_metrics.py -q`
- `uv run pytest -q`
- `uv run ruff check src/dltr/models/detection/metrics.py tests/detection/test_metrics.py`

# Next

- Keep non-training tests runnable under `dev` only.
- Reserve `torch` requirements for trainer smoke tests and runtime training paths.
