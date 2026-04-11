# 2026-04-10 08:45 端到端裁剪四点排序修复

## Summary

- 修复端到端文本框透视裁剪时四点顺序不稳定问题
- 在 `pipeline` 与 `end2end_system` 两处统一引入四点排序（TL/TR/BR/BL）
- 新增回归测试，验证不同点序输入得到一致裁剪结果

## Why

- 原逻辑在 polygon 已是 4 点时直接透传给 `cv2.getPerspectiveTransform`
- 当点顺序非 TL/TR/BR/BL 时，裁剪图会发生错位/翻转，导致识别误差异常放大

## Files Changed

- `src/dltr/pipeline/end_to_end.py`
- `src/dltr/models/end2end_system.py`
- `tests/pipeline/test_end_to_end.py`

## Verification

- `uv run ruff check src/dltr/pipeline/end_to_end.py src/dltr/models/end2end_system.py tests/pipeline/test_end_to_end.py`
- `uv run pytest tests/pipeline/test_end_to_end.py -q`
- 远端快速评估（300 张，`t=0.70, area=32`, detector+recognizer 组合）
  - `coverage=0.554786`
  - `system_acc=0.122166`
  - `matched_acc=0.220204`
  - `cer=0.566735`
