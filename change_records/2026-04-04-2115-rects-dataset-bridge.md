# Summary

Bridged the manually downloaded `ReCTS` dataset into the default project layout and added ReCTS-aware manifest/statistics handling.

# Why

The project needs to work with the real dataset layout already present in the repository, without forcing a full re-download or manual restructuring before any analysis can run.

# Files Changed

- `.gitignore`
- `src/dltr/data/inventory.py`
- `src/dltr/data/manifest.py`
- `tests/data/test_inventory_and_hardcase.py`
- `tests/data/test_manifest_and_reporting.py`

# Verification

- `uv run ruff check src/dltr/data tests/data`
- `uv run pytest tests/data -q`
- `uv run python -m dltr data build-rec-lmdb --config configs/data/datasets.example.yaml --dataset rects --output data/processed/rects_manifest.jsonl`
- `uv run python -m dltr data stats --config configs/data/datasets.example.yaml --output-name rects_first_eda.md`

# Next

- Add dataset-specific ReCTS crop extraction for detection/recognition splits.
- Ingest `ShopSign` or `LSVT` as the next complementary detection dataset.
