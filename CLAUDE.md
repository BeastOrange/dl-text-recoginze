# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese Scene Text Detection and Recognition project (`dltr` package) building an OCR pipeline with DBNet detection, CRNN/Transformer recognition, end-to-end inference, and post-OCR analysis. Python 3.11+, managed with `uv`.

## Commands

### Setup
```bash
uv python install 3.11
uv sync --extra dev                           # minimal dev
uv sync --extra dev --extra train-cu         # with CUDA training
uv sync --extra dev --extra train-cu --extra demo --extra viz  # full
```

### Setup（服务器仅国内镜像可达）
项目已在 `pyproject.toml` 的 `[[tool.uv.index]]` 中默认使用**清华 PyPI 镜像**，一般直接 `uv sync` 即可。若在境外需官方源：`UV_DEFAULT_INDEX=https://pypi.org/simple uv sync`。

勿用 `curl https://astral.sh/...` 安装 `uv`（常超时）。一键脚本（pip 从镜像装 uv + `uv sync`）：

```bash
cd /path/to/dl-text-recoginze
bash scripts/bootstrap_cn_server.sh
# 或指定镜像：UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ bash scripts/bootstrap_cn_server.sh
```

手动等价步骤：

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install -U uv -i "$UV_DEFAULT_INDEX"
cd /path/to/dl-text-recoginze && uv sync --extra dev --extra train-cu
```

说明：`uv python install 3.11` 会从官方源拉解释器，内网环境可改用机器已有 Python（如 Miniconda 3.11），只要 `uv sync` 时 `--python` 指过去即可。

### CLI Entry Point (preferred over `uv run dltr`)
```bash
uv run python scripts/run_dltr.py <command>
uv run python scripts/run_dltr.py --help     # show all command groups
```

### Common Commands
```bash
# Data
uv run python scripts/run_dltr.py data validate --config configs/data/datasets.example.yaml
uv run python scripts/run_dltr.py data prepare-detection --config configs/data/datasets.example.yaml --datasets rects shopsign
uv run python scripts/run_dltr.py data prepare-recognition --config configs/data/datasets.example.yaml --datasets rects shopsign

# Training
uv run python scripts/run_dltr.py train detector --config configs/detection/dbnet_baseline.yaml --run-id <id>
uv run python scripts/run_dltr.py train recognizer --config configs/recognition/transformer_baseline.yaml --run-id <id>
uv run python scripts/run_dltr.py train end2end --detector-config ... --recognizer-config ... --run-id <id>

# Evaluation (current semantics: writes provided metrics to report, not auto-inference)
uv run python scripts/run_dltr.py evaluate detector --config ... --precision 0.82 --recall 0.79 --hmean 0.80
uv run python scripts/run_dltr.py evaluate recognizer --run-name ... --word-accuracy 0.91
uv run python scripts/run_dltr.py evaluate end2end --image path/to/image.jpg --detector-run-dir ... --recognizer-run-dir ...
uv run python scripts/run_dltr.py evaluate end2end --manifest ... --sweep --sweep-detector-thresholds 0.3 0.4 0.5

# Reports
uv run python scripts/run_dltr.py report build-all --output-dir reports/train
uv run python scripts/run_dltr.py report summarize-english-benchmark --benchmark-jsons reports/eval/*_recognition_eval.json --output-dir reports/english

# Demo & Export
uv run python scripts/run_dltr.py demo --serve       # Streamlit
uv run python scripts/run_dltr.py demo --text "..."   # CLI demo
uv run python scripts/run_dltr.py export onnx --checkpoint ... --output ...

# CI checks
uv run pytest
uv run ruff check .
uv run python scripts/check_change_records.py
```

### English Benchmark
```bash
uv run --with scipy python scripts/run_dltr.py data prepare-recognition --config configs/data/datasets.english.example.yaml --datasets mjsynth iiit5k svt icdar13 icdar15
uv run --with easyocr python scripts/run_dltr.py evaluate recognizer-benchmark --run-name easyocr_iiit5k --backend easyocr --manifest ...
```

## Architecture

```
src/dltr/
├── cli.py / commands.py      # CLI entry: argument parsing + command routing
├── project.py                # Project root discovery (via PLAN.md)
├── data/                     # Dataset validation, manifest building, splits, charset
├── models/
│   ├── detection/            # DBNet: scaffold, dataset, trainer, inference, metrics, export
│   └── recognition/          # CRNN/Transformer-CTC: config, charset, dataset, trainer, inference, metrics, refinement
├── pipeline/                 # End-to-end: checkpoint discovery, E2E inference
├── visualization/            # Training plots, EDA, hard-case, ablation, english benchmark reports
├── post_ocr/                 # Rule-based scene text classification + structured field extraction
└── demo/                     # Streamlit app + runtime
```

Key data flow:
1. `data validate` → `data prepare-detection` / `data prepare-recognition` → splits + manifests
2. `train detector` / `train recognizer` → checkpoints in `artifacts/`
3. `evaluate end2end` → detection → crop → recognition → output
4. `report build-all` → aggregate training summaries under `reports/`

## Conventions

- **Python 3.11**, 4-space indent, Ruff 100-char line length
- `snake_case` for modules/functions, `PascalCase` for classes
- Explicit type hints on public logic
- Config files: YAML, named `{task}_{variant}.yaml` (e.g., `dbnet_baseline.yaml`, `transformer_4090.yaml`)
- Configs use frozen dataclasses; immutable patterns throughout
- Focused modules by domain (<800 lines, <50 line functions preferred)
- **Change records**: modifying `src/`, `tests/`, `configs/`, `scripts/`, `docs/`, `pyproject.toml`, `PLAN.md` requires a dated entry in `change_records/`

## Key Paths

| Path | Content |
|------|---------|
| `data/raw/` | Original downloaded datasets |
| `data/processed/` | Manifests, splits, charsets |
| `artifacts/detection/` | Detection checkpoints and summaries |
| `artifacts/checkpoints/recognition/` | Recognition checkpoints and summaries |
| `artifacts/end2end/` | End-to-end training artifacts |
| `reports/` | Training summaries, EDA, eval results, demo assets |
