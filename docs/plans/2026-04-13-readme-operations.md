# README Operations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the repository root README into an accurate operations handbook covering environment setup, dataset preparation, training, evaluation, and artifact locations.

**Architecture:** Keep all user-facing instructions in the root `README.md`, and add only minimal companion records under `docs/plans/` and `change_records/`. Reuse existing CLI behavior and config paths exactly as implemented so the document remains executable.

**Tech Stack:** Markdown, Python CLI (`scripts/run_dltr.py`), existing YAML configs, git

---

### Task 1: Capture the final README structure

**Files:**
- Modify: `README.md`
- Create: `docs/plans/2026-04-13-readme-operations-design.md`
- Create: `docs/plans/2026-04-13-readme-operations.md`

**Step 1: Write the target section outline**

Include:
- environment creation
- dataset layouts
- Chinese mainline data preparation
- training commands
- evaluation/report commands
- artifact locations
- English benchmark branch

**Step 2: Verify against existing CLI/configs**

Run:

```bash
uv run python scripts/run_dltr.py --help
```

Expected: root command groups match the README outline.

### Task 2: Rewrite the operations sections in README

**Files:**
- Modify: `README.md`

**Step 1: Replace sparse examples with runnable sequences**

Add exact command blocks for:
- `data validate`
- `data stats`
- `data prepare-detection`
- `data prepare-recognition`
- `data prepare-recognition-crops`
- `train detector`
- `train recognizer`
- `train end2end`
- `evaluate end2end`
- `report build-all`

**Step 2: Clarify semantics**

Explicitly document:
- metric archival vs true inference
- optional extras (`demo`, `viz`, `train-cu`, `scipy`, `easyocr`)
- default output directories

### Task 3: Record the documentation change

**Files:**
- Create: `change_records/2026-04-13-readme-operations-handbook.md`

**Step 1: Summarize what changed**

Mention:
- README upgraded to an operations handbook
- added full Chinese mainline workflow
- clarified output directories and evaluation semantics

### Task 4: Verify, commit, and push

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-04-13-readme-operations-design.md`
- Modify: `docs/plans/2026-04-13-readme-operations.md`
- Modify: `change_records/2026-04-13-readme-operations-handbook.md`

**Step 1: Run verification**

Run:

```bash
uv run python scripts/run_dltr.py --help
uv run python scripts/run_dltr.py data validate --config configs/data/datasets.english.local.yaml
uv run ruff check README.md
```

Expected:
- help command succeeds
- local benchmark config validates
- ruff check succeeds

**Step 2: Commit**

```bash
git add README.md docs/plans/2026-04-13-readme-operations-design.md docs/plans/2026-04-13-readme-operations.md change_records/2026-04-13-readme-operations-handbook.md
git commit -m "docs(readme): 补充完整运行手册"
```

**Step 3: Push**

```bash
git push origin main
```
