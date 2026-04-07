# 2026-04-07 00:15 检测兼容性与报告过滤修复

## Summary

- 修复检测推理阶段的 BGR / RGB 颜色空间不一致
- 修复旧 detector checkpoint 缺失 `model_architecture` 元数据时的加载兼容性
- 收窄训练汇总中的 obsolete 过滤规则，只排除 `report-smoke`
- 补强训练汇总测试断言，并恢复检测训练测试中的图像 helper

## Why

- 代码审查发现检测训练与推理并未真正统一预处理
- 历史 detector 权重会被误按新结构加载
- 训练汇总函数错误地把合法扩展 run 当作 obsolete
- 测试缺失关键断言，导致回归问题可能被放过

## Files Changed

- `src/dltr/models/detection/inference.py`
- `src/dltr/visualization/training_reports.py`
- `tests/detection/test_inference.py`
- `tests/detection/test_trainer.py`
- `tests/visualization/test_training_reports.py`
- `tests/visualization/test_training_reports_filters.py`

## Verification

- `uv run ruff check src tests`
- `uv run pytest -q`

## Next

- 将这轮修复同步回服务器工作目录
- 若继续让 Claude 在 worktree 开发，需要先清理主工作区脏改动，避免再次越界
