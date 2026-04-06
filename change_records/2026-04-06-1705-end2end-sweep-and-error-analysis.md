# 2026-04-06 17:05 系统级 sweep 与错误分析报告

## Summary

- 为 `evaluate end2end --manifest` 增加 `--sweep` 模式
- 新增可配置的 `--sweep-detector-thresholds` 与 `--sweep-min-areas`
- 普通 baseline 评估自动生成固定文件名错误分析报告
- 输出固定产物：
  - `end2end_sweep_summary.json/.md`
  - `end2end_error_analysis.json/.md`

## Why

- 当前系统级 baseline 已经跑通，但还缺少参数扫描能力
- 需要快速判断 detection threshold / min_area 是否有白拿收益
- 需要结构化错误分析，区分 missed detection 与 wrong recognition

## Files Changed

- `src/dltr/cli.py`
- `src/dltr/commands.py`
- `src/dltr/pipeline/end_to_end_baseline.py`
- `tests/test_end_to_end_command.py`
- `tests/pipeline/test_end_to_end_baseline.py`

## Verification

- `uv run ruff check .`
- `uv run pytest -q`

## Next

- 在服务器上执行系统级 sweep
- 读取错误分析报告并决定优先优化 detection 还是 recognition
