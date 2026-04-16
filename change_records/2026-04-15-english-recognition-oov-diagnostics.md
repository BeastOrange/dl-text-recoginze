# 2026-04-15 English Recognition OOV Diagnostics

- Added pre-training recognition diagnostics to expose dataset composition and charset coverage:
  - New file: `src/dltr/models/recognition/diagnostics.py`
  - Trainer now writes:
    - `training_diagnostics.json`
    - `training_diagnostics.md`
  - Diagnostics include sample counts, dataset counts, empty-label counts, OOV ratio, and top OOV chars.
- Added OOV guardrail for recognition training:
  - New config fields:
    - `max_oov_ratio` (optional, fail-fast threshold)
    - `diagnostics_top_k` (default 20)
  - If `train` OOV ratio exceeds `max_oov_ratio`, training stops early with an explicit error.
- Wired new config fields into training plan artifacts in `src/dltr/commands.py`.
- Extended charset utility with `encode_with_oov_count` to support explicit unknown-character accounting.
- Added regression tests:
  - `tests/recognition/test_charset.py`: OOV count behavior.
  - `tests/recognition/test_config.py`: new config field parsing.
  - `tests/recognition/test_trainer.py`: diagnostics artifact generation + OOV threshold failure.

## Verification

- `uv run ruff check src/dltr/models/recognition tests/recognition src/dltr/commands.py`
- `uv run pytest tests/recognition/test_charset.py tests/recognition/test_config.py tests/recognition/test_trainer.py -q`
