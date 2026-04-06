# 2026-04-06 00:35 主报告重构为 OCR 主线

## Summary

- 将训练主报告目录收口为 OCR 主线
- 语义训练汇总迁移为扩展模块报告
- Streamlit 前端新增扩展报告区域

## Why

- 项目总方案已重构为“检测识别主线 + OCR 后规则理解扩展”
- 若主报告目录仍默认汇总语义训练结果，会继续造成口径冲突
- 需要让训练报告、项目汇总和前端展示与新的总方案一致

## Files Changed

- `src/dltr/commands.py`
- `src/dltr/visualization/project_summary.py`
- `src/dltr/visualization/report_index.py`
- `src/dltr/visualization/ablation_reports.py`
- `src/dltr/demo/streamlit_app.py`
- `tests/test_report_build_all_command.py`
- `tests/demo/test_streamlit_app.py`

## Verification

- `uv run ruff check .`
- `uv run pytest -q`

## Next

- 继续将语义相关内容从主结论页、主汇总页逐步降级为扩展或历史 baseline
- 进入检测与识别正式实验阶段
