# Summary

Localized the Streamlit frontend chrome and interaction copy from English to Simplified Chinese while keeping report artifacts unchanged.

# Why

The project frontend is intended for Chinese-speaking thesis demos, so the UI shell should use Chinese even though experimental report files remain in English for paper and slide reuse.

# Files Changed

- `src/dltr/demo/streamlit_app.py`

# Verification

- `uv run pytest tests/demo/test_streamlit_app.py tests/test_demo_command.py -q`
- `uv run ruff check src/dltr/demo/streamlit_app.py`

# Next

- Keep the Streamlit chrome in Chinese
- Keep report artifact bodies in English unless explicitly changing the reporting spec
