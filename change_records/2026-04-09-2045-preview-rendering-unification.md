# 2026-04-09 20:45 预览图统一渲染与标签重叠修复

## Summary

- 新增共享端到端预览渲染模块
- 端到端预览图改为“图内短标签 + 右侧图例栏”
- 为标签增加背景块与基础避让逻辑
- 分离式端到端与统一多任务端到端统一使用同一套渲染逻辑

## Why

- 当前预览图标签直接堆叠在检测框附近，密集场景重叠严重
- 可视化样式粗糙，不适合客户交付和演示
- 两条端到端链路各自维护绘制逻辑，风格不一致

## Files Changed

- `src/dltr/visualization/end_to_end_rendering.py`
- `src/dltr/pipeline/end_to_end.py`
- `src/dltr/models/end2end_system.py`
- `tests/visualization/test_end_to_end_rendering.py`
- `tests/pipeline/test_end_to_end.py`
- `docs/plans/2026-04-09-visualization-rendering-design.md`
- `docs/plans/2026-04-09-visualization-rendering.md`

## Verification

- `uv run pytest tests/visualization/test_end_to_end_rendering.py tests/pipeline/test_end_to_end.py -q`
- `uv run pytest -q`
- `uv run ruff check .`
