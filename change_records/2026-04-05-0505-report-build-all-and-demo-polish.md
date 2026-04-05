# Summary

Completed the report-suite automation with `report build-all`, added training report discovery to the Streamlit demo, and polished the demo runtime flow for static and served modes.

# Why

The project already had individual report commands, but it still lacked a one-shot way to assemble a thesis-ready report set and a demo view that could actually consume those generated outputs.

# Files Changed

- `src/dltr/commands.py`
- `src/dltr/cli.py`
- `src/dltr/demo/streamlit_app.py`
- `src/dltr/pipeline/checkpoints.py`
- `src/dltr/pipeline/__init__.py`
- `tests/demo/test_streamlit_app.py`
- `tests/test_report_build_all_command.py`

# Verification

- `uv run pytest -q`
- `uv run ruff check .`

# Next

- Generate real `reports/train/` assets from actual runs and surface them in the Streamlit UI.
- Add task-level quick actions for “latest detection run” and “latest recognition run” inside the frontend.
