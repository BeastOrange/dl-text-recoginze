# Summary

Added an English Streamlit demo frontend and split the demo command into interactive UI mode and static asset generation mode.

# Why

The project needed a real user-facing frontend instead of only CLI outputs and markdown files. Streamlit is a practical fit for a thesis demo because it can directly render the existing reports and pipeline outputs.

# Files Changed

- `src/dltr/demo/__init__.py`
- `src/dltr/demo/streamlit_app.py`
- `src/dltr/commands.py`
- `src/dltr/cli.py`
- `tests/demo/test_streamlit_app.py`
- `tests/test_demo_command.py`

# Verification

- `uv sync --extra demo`
- `uv run pytest tests/demo/test_streamlit_app.py tests/test_demo_command.py -q`

# Next

- Add image upload and direct end-to-end inference trigger inside the Streamlit app
- Show training summary charts inline in the UI instead of markdown only
