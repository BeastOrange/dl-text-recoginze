# 2026-04-08 02:28 CI 训练测试在无 Torch 环境下自动跳过

## Summary

- 为 `tests/test_end2end_training_command.py` 增加 `pytest.importorskip("torch")`

## Why

- GitHub CI 只安装 `dev` 依赖，不安装 `train-cu`
- 新增的端到端训练续训测试会真实导入训练模块
- 在无 `torch` 环境下应与其他训练测试保持一致，自动跳过而不是失败

## Files Changed

- `tests/test_end2end_training_command.py`

## Verification

- `uv run pytest tests/test_end2end_training_command.py -q`
- `uv run pytest -q`
- `uv run ruff check .`
