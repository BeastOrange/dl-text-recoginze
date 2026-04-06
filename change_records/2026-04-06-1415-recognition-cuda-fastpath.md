# 2026-04-06 14:15 识别训练 CUDA 快路径优化

## Summary

- 为识别训练新增 CUDA 快路径优化，包括 `pin_memory`、`persistent_workers`、`prefetch_factor`
- 训练阶段启用 `non_blocking` 拷贝、AMP、GradScaler 与 TF32
- 新增 `transformer_4090.yaml` 作为 4090 识别训练专用配置
- 补充运行时优化与 4090 配置的测试覆盖

## Why

- 识别训练阶段 GPU 利用率明显低于检测训练阶段
- 当前识别 dataloader 使用 `num_workers: 0`，且未开启常见 CUDA 数据传输优化
- 需要先用低风险方案提升吞吐，保留后续再做数据解码/缓存层优化的空间

## Files Changed

- `src/dltr/models/recognition/trainer.py`
- `configs/recognition/transformer_4090.yaml`
- `tests/recognition/test_trainer.py`
- `tests/recognition/test_config.py`

## Verification

- `uv run pytest tests/recognition/test_trainer.py tests/recognition/test_config.py -q`
- `uv run ruff check .`
- `uv run pytest -q`

## Next

- 若 4090 上识别训练 GPU 利用率仍偏低，再进入数据解码与缓存层优化
