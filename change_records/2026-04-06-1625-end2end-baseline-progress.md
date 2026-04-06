# 2026-04-06 16:25 系统级 baseline 评估增加进度输出

## Summary

- 为 `evaluate end2end --manifest` 批量评估模式接入 `ProgressBar`
- 每处理一张图更新一次累计指标，显示 `images`、`matched`、`coverage`、`exact`
- 增加对应的进度输出测试覆盖

## Why

- 当前系统级 baseline 评估前台执行时长时间无输出，容易误判为卡死
- 需要让服务器前台直接执行命令时能看到实时进度，而不依赖 `tail`

## Files Changed

- `src/dltr/pipeline/end_to_end_baseline.py`
- `tests/pipeline/test_end_to_end_baseline.py`

## Verification

- `uv run ruff check .`
- `uv run pytest -q`

## Next

- 在服务器上重新运行完整 val 集系统级 baseline
