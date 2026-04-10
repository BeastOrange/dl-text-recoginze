# 2026-04-09 23:05 报告可视化防重叠与样式统一

## Summary

- 修复 `reports/train` 训练汇总图在长 run 名场景下的文字重叠
- 统一训练汇总、hardcase、ablation 图的配色、网格、边框和数值标注样式
- 修复 `end_to_end_preview` 中文文本渲染问号问题（支持 CJK 字体探测与 PIL 绘制）
- hardcase 全 0 场景增加显式提示与 0 值标注，避免“空白图”误判
- 远端重跑 `report build-all` 与 `evaluate end2end --image`，并回传覆盖本地标准目录
- 清理本地与远端 `reports/*_tonight*` 目录，仅保留标准目录

## Why

- 既有可视化在长标签和多 run 场景下可读性差，影响交付质量
- 历史后缀目录导致报告路径不统一，影响客户交付与后续维护

## Files Changed

- `src/dltr/visualization/training_reports.py`
- `src/dltr/visualization/hardcase_reports.py`
- `src/dltr/visualization/ablation_reports.py`
- `src/dltr/visualization/end_to_end_rendering.py`
- `src/dltr/visualization/plot_style.py`
- `tests/visualization/test_end_to_end_rendering.py`
- `tests/visualization/test_training_reports.py`
- `tests/visualization/test_hardcase_reports.py`
- `tests/visualization/test_ablation_reports.py`

## Verification

- `uv run ruff check src/dltr/visualization tests/visualization`
- `uv run pytest tests/visualization tests/test_report_build_all_command.py tests/test_report_command.py tests/test_project_report_command.py -q`
- `uv run pytest tests/visualization/test_end_to_end_rendering.py tests/visualization/test_hardcase_reports.py -q`
- 远端：`uv sync --extra dev` 后 `report build-all --output-dir reports/train`
- 远端：`uv sync --extra train-cu` 后 `evaluate end2end --image ... --output-dir reports/eval`
- 远端：`apt-get install -y fonts-noto-cjk`，确认 `NotoSansCJK-Regular.ttc` 可用
- 本地验收：`reports/train/*.png`、`reports/hardcase/hardcase_overview.png`、`reports/ablation/ablation_overview.png`、`reports/eval/end_to_end_preview.png`
