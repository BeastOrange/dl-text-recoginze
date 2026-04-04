# Summary

Added infrastructure automation for CI enforcement, Linux sync, and Windows bootstrap.

# Why

The project requires strict process governance for frequent GitHub updates, change-record discipline, and cross-platform execution.

# Files Changed

- `.github/workflows/ci.yml`
- `scripts/check_change_records.py`
- `scripts/sync_to_linux.sh`
- `scripts/bootstrap_windows.ps1`

# Verification

- `python3 scripts/check_change_records.py`
- `python3 -m py_compile scripts/check_change_records.py`
- `bash -n scripts/sync_to_linux.sh`
- `uv run ruff check scripts/check_change_records.py`

# Next

Wire CI with repository branch protections and run first remote CI pass after initial push.
