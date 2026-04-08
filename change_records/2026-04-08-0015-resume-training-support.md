# 2026-04-08 00:15 训练断点续训支持

## Summary

- 为 `train detector` / `train recognizer` / `train end2end` 增加 `--resume-from`
- 统一支持从 checkpoint 文件或 run 目录恢复训练
- 恢复模型参数、优化器状态、AMP scaler 状态、epoch 与已有训练历史
- 续训时保持原 `run_dir`，继续更新 `best.pt`、`last.pt`、`training_history.jsonl`

## Why

- 训练中断后无法继续，重复从头训练成本过高
- 服务器训练周期长，断点续训是必要能力
- 三条训练链路需要统一 resume 协议，便于命令使用和后续维护

## Files Changed

- `src/dltr/cli.py`
- `src/dltr/commands.py`
- `src/dltr/models/detection/trainer.py`
- `src/dltr/models/recognition/trainer.py`
- `src/dltr/models/end2end_system.py`
- `tests/detection/test_trainer.py`
- `tests/recognition/test_trainer.py`
- `tests/test_end2end_training_command.py`
- `tests/test_cli.py`
- `tests/test_commands.py`

## Verification

- `uv run pytest tests/detection/test_trainer.py tests/recognition/test_trainer.py tests/test_end2end_training_command.py tests/test_cli.py tests/test_commands.py -q`
- `uv run pytest -q`
- `uv run ruff check .`
