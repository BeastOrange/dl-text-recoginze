# 2026-04-06 16:45 系统级 baseline 评估快路径

## Summary

- detection inference 新增可复用的 `DetectionPredictorSession`
- recognition inference 新增可复用的 `RecognitionPredictorSession` 与内存图像识别接口
- end-to-end pipeline 支持注入 detector/recognizer session，并改为内存 crop 推理
- manifest baseline 评估改为单次加载模型、整轮复用，不再重复 `torch.load` 和重建模型

## Why

- 完整 val 集系统级 baseline 评估速度过慢
- 原实现对每张图都会重复加载 detector/recognizer checkpoint，并对每个 crop 落盘后再读回
- CPU/I/O 成为瓶颈，GPU 利用率过低

## Files Changed

- `src/dltr/models/detection/inference.py`
- `src/dltr/models/recognition/inference.py`
- `src/dltr/pipeline/end_to_end.py`
- `src/dltr/pipeline/end_to_end_baseline.py`
- `tests/pipeline/test_end_to_end.py`
- `tests/pipeline/test_end_to_end_baseline.py`

## Verification

- `uv run ruff check .`
- `uv run pytest -q`

## Next

- 在服务器上重新运行完整 val 集系统级 baseline，观察吞吐是否显著改善
