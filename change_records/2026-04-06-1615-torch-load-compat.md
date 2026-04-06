# 2026-04-06 16:15 远端 PyTorch checkpoint 加载兼容修复

## Summary

- 新增统一的 `torch checkpoint` 加载辅助函数
- 显式使用 `weights_only=False` 加载受信任训练产物
- 对旧版不支持 `weights_only` 参数的 PyTorch 自动回退
- 修复 detection inference、recognition inference 与 second-pass policy 的 checkpoint 读取路径
- 增加兼容性测试覆盖

## Why

- 服务器上的 PyTorch 2.8 默认开启 `weights_only=True`
- 现有 checkpoint 配置中包含 `pathlib.PosixPath`，导致系统级 baseline 评估时加载失败
- 需要在不破坏旧版 PyTorch 兼容性的前提下恢复推理/评估能力

## Files Changed

- `src/dltr/torch_checkpoint.py`
- `src/dltr/models/detection/inference.py`
- `src/dltr/models/recognition/inference.py`
- `src/dltr/pipeline/end_to_end.py`
- `tests/test_torch_checkpoint.py`

## Verification

- `uv run pytest tests/test_torch_checkpoint.py tests/test_end_to_end_command.py -q`
- `uv run ruff check .`
- `uv run pytest -q`

## Next

- 重新在服务器上执行系统级 baseline 评估命令
