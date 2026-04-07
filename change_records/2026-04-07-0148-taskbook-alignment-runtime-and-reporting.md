# 2026-04-07 01:48 任务书对齐补齐与系统指标收口

## Summary

- 为 `train end2end` 补充系统实验变体元数据与联合编排说明
- 为端到端推理与系统评估补充 runtime metrics，包括总延迟、阶段延迟与 FPS
- 收口训练报告索引与 Streamlit 训练报告展示，隐藏 legacy `semantic_*` 主线残留
- 重生成主线训练汇总与端到端预览产物，统一使用 `analysis_label`

## Why

- 当前仓库实现与任务书主线基本一致，但系统级指标、实验口径和样例产物仍有偏差
- legacy `semantic_*` 报告仍会污染主线展示，影响答辩口径
- 端到端系统缺少可量化的实时性证据，不足以支撑“可实时处理”的表述
- `train end2end` 需要输出更明确的系统实验元数据，便于将实现映射到任务书实验矩阵

## Files Changed

- `src/dltr/commands.py`
- `src/dltr/demo/streamlit_app.py`
- `src/dltr/pipeline/end_to_end.py`
- `src/dltr/pipeline/end_to_end_baseline.py`
- `src/dltr/visualization/report_index.py`
- `tests/demo/test_streamlit_app.py`
- `tests/pipeline/test_end_to_end.py`
- `tests/pipeline/test_end_to_end_baseline.py`
- `tests/test_end2end_training_command.py`
- `tests/visualization/test_report_index.py`
- `reports/train/`
- `reports/eval/end2end_preview.json`
- `reports/demo_assets/demo_preview_analysis_report.md`

## Verification

- `uv run pytest tests/visualization/test_report_index.py tests/demo/test_streamlit_app.py tests/test_end2end_training_command.py tests/pipeline/test_end_to_end.py tests/pipeline/test_end_to_end_baseline.py -q`
- `uv run pytest tests/test_commands.py tests/test_report_build_all_command.py tests/visualization/test_project_summary.py tests/test_end_to_end_command.py -q`
- `uv run python scripts/run_dltr.py report build-all --output-dir reports/train`
- `uv run python scripts/run_dltr.py evaluate end2end --text "营业时间09:00-21:00" --confidence 0.52 --blur-score 0.6 --output reports/eval/end2end_preview.json`
- `uv run python scripts/run_dltr.py demo --text "当心高压 电话13800138000" --output-dir reports/demo_assets`

## Next

- 若后续继续补齐答辩材料，可基于新的 runtime metrics 增补系统级延迟/FPS 图表
- 若需要进一步逼近任务书“弯曲文本”要求，可在 hard-case 报告中追加曲线文本失败案例专题
