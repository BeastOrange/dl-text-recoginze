# 2026-04-06 17:35 detector-aware 识别微调与 improved detector 配置

## Summary

- `prepare-recognition-crops` 默认输出切换为独立的 crop 版识别数据路径
- 新增 `transformer_detector_crop_4090.yaml`
- 新增 `dbnet_improved_4090.yaml`
- detection trainer 中让 `hard_case_sampling` 与 `multi_scale_augmentation` 真正生效
- 增加对应配置与训练行为测试

## Why

- 当前系统级 OCR 掉点的核心是 detector crop 分布与 recognizer 训练分布失配
- 需要单独产出 detector-aware 的识别训练数据与配置，避免污染原 baseline
- detection 改进配置中的 hard-case / multi-scale 不能继续停留在占位状态

## Files Changed

- `src/dltr/commands.py`
- `src/dltr/models/detection/trainer.py`
- `configs/recognition/transformer_detector_crop_4090.yaml`
- `configs/detection/dbnet_improved_4090.yaml`
- `tests/detection/test_trainer.py`
- `tests/recognition/test_config.py`
- `tests/test_recognition_crop_command.py`

## Verification

- `uv run ruff check .`
- `uv run pytest -q`

## Next

- 同步服务器并执行 detector-crop 识别微调
- 执行 improved detector 训练
- 重新跑系统级 baseline / sweep / 错误分析
