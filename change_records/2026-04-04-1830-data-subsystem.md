# Summary

Implemented a data subsystem scaffold for Chinese scene-text project setup, including path validation, inventory/statistics, hard-case metadata heuristics, recognition manifest generation, and English markdown EDA reporting.

# Why

The project needs a robust and testable data layer before model training. This change provides deterministic tooling that works even when datasets are partially available or missing.

# Files Changed

- `src/dltr/data/__init__.py`
- `src/dltr/data/types.py`
- `src/dltr/data/config.py`
- `src/dltr/data/validation.py`
- `src/dltr/data/inventory.py`
- `src/dltr/data/hardcase.py`
- `src/dltr/data/manifest.py`
- `src/dltr/data/reporting.py`
- `src/dltr/visualization/__init__.py`
- `src/dltr/visualization/eda_markdown.py`
- `configs/data/datasets.example.yaml`
- `configs/data/hardcase_rules.example.yaml`
- `reports/eda/README.md`
- `tests/data/test_config.py`
- `tests/data/test_validation.py`
- `tests/data/test_inventory_and_hardcase.py`
- `tests/data/test_manifest_and_reporting.py`

# Verification

- `uv run pytest tests/data -q`

# Next

- Wire these public functions into CLI commands (`data validate`, `data stats`, and report generation command).
- Add dataset-specific parsers for RCTW-17/ReCTS label schemas when real data is available.
