# Summary

Removed generated packaging metadata from version control and added an ignore rule for future builds.

# Why

`*.egg-info` is generated build metadata and should not be committed. Keeping it out of the repository reduces noise and avoids accidental diff churn.

# Files Changed

- `.gitignore`
- `src/dltr.egg-info/` (removed from git tracking)

# Verification

- `git status --short`

# Next

- Keep generated build metadata out of commits and continue using `uv`/`setuptools` for local packaging.
