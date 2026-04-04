# Summary

Integrated the local `ShopSign_1265` dataset layout into the project data subsystem and documented the standard dataset bridge layout.

# Why

The downloaded ShopSign sample set uses a layout that differs from ReCTS and from the generic same-folder image/label assumption. The project needs explicit handling so data commands work directly on the downloaded files.

# Files Changed

- `.gitignore`
- `src/dltr/data/inventory.py`
- `src/dltr/data/manifest.py`
- `tests/data/test_inventory_and_hardcase.py`
- `tests/data/test_manifest_and_reporting.py`
- `data/README.md`

# Verification

- `uv run ruff check src/dltr/data tests/data`
- `uv run pytest tests/data -q`
- `uv run python -m dltr data build-rec-lmdb --config configs/data/datasets.example.yaml --dataset shopsign --output data/processed/shopsign_manifest.jsonl`

# Next

- Bridge `ShopSign_1265` into `data/raw/shopsign`
- Add ShopSign-aware split preparation for detection and recognition experiments
