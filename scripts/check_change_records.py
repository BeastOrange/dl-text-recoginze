#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRACKED_PREFIXES = (
    "src/",
    "tests/",
    "configs/",
    "scripts/",
    "docs/",
)

TRACKED_FILES = {
    "pyproject.toml",
    "uv.lock",
    "PLAN.md",
}


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _has_commit(ref: str) -> bool:
    result = _run_git(["rev-parse", "--verify", "--quiet", ref])
    return result.returncode == 0


def _resolve_base() -> str | None:
    # Priority 1: CI-provided explicit base SHA.
    for env_name in ("BASE_SHA", "GITHUB_BASE_SHA"):
        value = os.getenv(env_name, "").strip()
        if value and _has_commit(value):
            return value

    # Priority 2: local development fallback.
    if _has_commit("HEAD~1"):
        return "HEAD~1"
    return None


def _changed_files(base: str) -> list[str]:
    result = _run_git(["diff", "--name-only", f"{base}..HEAD"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _touches_tracked_area(path: str) -> bool:
    return path in TRACKED_FILES or path.startswith(TRACKED_PREFIXES)


def main() -> int:
    base = _resolve_base()
    if base is None:
        print(
            "No BASE_SHA and no HEAD~1 available; skipping change_records enforcement.",
        )
        return 0

    changed = _changed_files(base)
    if not changed:
        print(f"No changed files found for range {base}..HEAD.")
        return 0

    touches_tracked = any(_touches_tracked_area(path) for path in changed)
    has_change_record = any(path.startswith("change_records/") for path in changed)

    if touches_tracked and not has_change_record:
        print("Detected changes in code/config/scripts/docs but no change_records file.")
        print(f"Compared range: {base}..HEAD")
        print("Changed files:")
        for path in changed:
            print(f"  - {path}")
        return 1

    print("change_records check passed.")
    print(f"Compared range: {base}..HEAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
