# 2026-04-06 16:05 新增系统级 baseline 评估命令

## Summary

- 扩展 `evaluate end2end`，新增基于 `detection_splits` manifest 的批量评估模式
- 新增系统级 baseline 评估模块，支持 polygon IoU 匹配、批量聚合指标与固定文件名 summary
- 输出系统级核心指标：`detection_coverage`、`matched_line_accuracy`、`system_line_accuracy`、`matched_cer`、`matched_ned`
- 增加命令级与评估模块级测试覆盖

## Why

- 当前仓库只能做单图端到端推理，无法对整套验证集形成系统级 baseline
- 下一阶段需要用正式的系统级指标判断瓶颈到底在检测还是识别
- 需要一个可重复执行、固定输出路径的评估入口，方便服务器直接跑结果

## Files Changed

- `src/dltr/cli.py`
- `src/dltr/commands.py`
- `src/dltr/pipeline/__init__.py`
- `src/dltr/pipeline/end_to_end_baseline.py`
- `tests/test_end_to_end_command.py`
- `tests/pipeline/test_end_to_end_baseline.py`

## Verification

- `uv run pytest tests/test_end_to_end_command.py tests/pipeline/test_end_to_end_baseline.py -q`
- `uv run ruff check .`
- `uv run pytest -q`

## Next

- 在服务器上运行完整 val 集的系统级 baseline
- 根据 baseline 结果决定优先优化 detection recall 还是 recognition 精调
