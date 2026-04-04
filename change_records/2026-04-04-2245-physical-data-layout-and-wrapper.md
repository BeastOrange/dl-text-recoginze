# Summary

Switched the runtime entrypoint to an explicit wrapper script and reorganized the downloaded datasets into a clean physical layout under `data/raw/`.

# Why

The previous startup approach relied on implicit Python path behavior and the downloaded datasets were split between project root and bridge paths. This change makes runtime behavior explicit and keeps the repository structure consistent.

# Files Changed

- `scripts/run_dltr.py`
- `scripts/bootstrap_windows.ps1`
- `src/dltr/models/detection/scaffold.py`
- `PLAN.md`
- `data/README.md`
- `src/dltr/data/inventory.py`
- `src/dltr/data/manifest.py`
- `tests/test_commands.py`
- `tests/data/test_inventory_and_hardcase.py`
- `tests/data/test_manifest_and_reporting.py`
- `.gitignore`
- `change_records/2026-04-04-2205-shopsign-bridge.md`

# Verification

- `uv sync --extra dev`
- `uv run pytest -q`
- `uv run ruff check .`
- `uv run python scripts/run_dltr.py data build-rec-lmdb --config configs/data/datasets.example.yaml --dataset rects --output data/processed/rects_manifest.jsonl`
- `uv run python scripts/run_dltr.py data build-rec-lmdb --config configs/data/datasets.example.yaml --dataset shopsign --output data/processed/shopsign_manifest.jsonl`
- `uv run python scripts/check_change_records.py`

# Next

- Add dataset-specific split preparation for ReCTS train/test and ShopSign subsets.
- Continue with real detection/recognition training pipeline implementation on top of the cleaned data layout.
