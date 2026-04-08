# 2026-04-07 11:20 Detector AMP 与 DataLoader Fastpath

## Summary

- 为 detector 训练新增 CUDA runtime 优化配置
- 在 detector DataLoader 中启用 `pin_memory`
- 在 `num_workers > 0` 时启用 `persistent_workers` 与 `prefetch_factor`
- 为 detector 前向训练与验证接入 CUDA AMP 与 `GradScaler`
- 为 detector tensor 搬运启用 `non_blocking`

## Why

- 4090 上 detector 训练吞吐偏低，单轮时间过长
- recognizer 已有更成熟的 CUDA fastpath，detector 需要对齐
- 在不改模型结构的前提下，runtime 优化是最直接的提速手段

## Files Changed

- `src/dltr/models/detection/trainer.py`
- `tests/detection/test_trainer.py`
- `docs/plans/2026-04-07-detector-runtime-optimization.md`

## Verification

- `uv run pytest tests/detection/test_trainer.py -q`
- `uv run pytest -q`
- `uv run ruff check .`
