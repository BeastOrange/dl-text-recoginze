# 2026-04-06 00:10 总方案重构为 OCR 主线

## Summary

- 重写 `PLAN.md`
- 将项目总口径从“检测/识别/语义并列”改为“检测识别主线 + OCR 后规则理解扩展”
- 同步调整项目描述、CLI 描述和 Streamlit 首页文案

## Why

- 现有方案把语义写成了并列主任务，和项目真实定位冲突
- 当前可稳定复现、可答辩的主线应是中文自然场景文本检测与识别
- OCR 数据集原生支撑检测与识别，不天然支撑语义监督训练
- 需要在不丢失老师“加入理解能力”要求的前提下，重构总方案叙事

## Files Changed

- `PLAN.md`
- `pyproject.toml`
- `src/dltr/cli.py`
- `src/dltr/demo/streamlit_app.py`

## Verification

- `uv run ruff check .`
- `uv run pytest -q`

## Next

- 按新的主线口径继续完善检测与识别实验
- 将语义相关代码和报告逐步降级为扩展模块或历史 baseline
