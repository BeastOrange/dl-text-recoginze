# 2026-04-07 10:38 任务书缺口补齐与统一多任务主线落盘

## Summary

- 新增统一多任务训练/推理模型，`train end2end` 改为真实共享主干多任务训练
- `evaluate end2end` 新增统一 checkpoint / run dir 支持，并保留旧双 checkpoint 兼容
- 检测、裁切、端到端评估链路从固定四点框扩展到通用偶数点 polygon
- `export onnx` 从生成计划改为真实导出逻辑，并补明确依赖报错
- 新增本地快速配置，基于公开数据裁切样本生成真实主线训练与报告产物

## Why

- 之前的 `train end2end` 只是检测与识别分开训练后汇总，不能支撑“多任务协同训练”的任务书口径
- 识别主线默认数据规模过小，导致公开数据实验结果长期为空
- 曲线/非四点文本缺少代码证据，影响“复杂自然场景文本”要求
- ONNX 导出停留在计划层，不能支撑轻量化/部署优化口径

## Files Changed

- `src/dltr/cli.py`
- `src/dltr/commands.py`
- `src/dltr/models/end2end_system.py`
- `src/dltr/models/detection/export.py`
- `src/dltr/models/detection/dataset.py`
- `src/dltr/models/recognition/trainer.py`
- `src/dltr/data/detection_preparation.py`
- `src/dltr/data/recognition_crops.py`
- `src/dltr/pipeline/end_to_end.py`
- `src/dltr/pipeline/end_to_end_baseline.py`
- `configs/detection/dbnet_multitask_quick.yaml`
- `configs/recognition/transformer_detector_crop_quick.yaml`
- `docs/plans/2026-04-07-taskbook-gap-closure-design.md`
- `docs/plans/2026-04-07-taskbook-gap-closure.md`
- `tests/test_end2end_training_command.py`
- `tests/test_end_to_end_command.py`
- `tests/test_commands.py`
- `tests/pipeline/test_end_to_end.py`
- `tests/pipeline/test_end_to_end_baseline.py`
- `tests/data/test_detection_preparation.py`
- `tests/data/test_recognition_crops.py`
- `artifacts/end2end/taskbook-mainline/`
- `reports/train/`
- `reports/eval/`

## Verification

- `uv run pytest -q`
- `uv run ruff check .`
- `uv run python scripts/run_dltr.py data prepare-recognition-crops --max-samples 512`
- `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/run_dltr.py train end2end --detector-config configs/detection/dbnet_multitask_quick.yaml --recognizer-config configs/recognition/transformer_detector_crop_quick.yaml --run-id taskbook-mainline --max-train-batches 8 --max-val-batches 2`
- `uv run python scripts/run_dltr.py report build-all --output-dir reports/train`
- `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/run_dltr.py evaluate end2end --image data/raw/shopsign/images/image_15.jpg --end2end-run-dir artifacts/end2end/taskbook-mainline --output-dir reports/eval`
- `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/run_dltr.py evaluate end2end --manifest data/processed/detection_splits/val.jsonl --end2end-run-dir artifacts/end2end/taskbook-mainline --max-images 20 --output-dir reports/eval`
- `.venv/bin/python scripts/run_dltr.py export onnx --config configs/detection/dbnet_multitask_quick.yaml --checkpoint artifacts/tmp_export/det_dummy.pt --output artifacts/tmp_export/det_dummy.onnx`

## Notes

- 本次主线实验为“受控快速验证”配置，核心目标是把统一多任务训练、统一推理和公开数据主线结果真正跑通并落盘
- 在 MPS 上训练识别 CTC 分支时启用了 fallback，提示仍会出现，但命令可完成
- `uv run` 环境未直接包含临时安装的 ONNX 依赖，因此真实 ONNX 导出验证使用了当前 `.venv/bin/python`
