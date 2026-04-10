# 可视化渲染统一与重叠修复 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 统一端到端结果图的渲染风格，修复标签重叠，并让输出更适合客户交付。

**Architecture:** 新增共享渲染模块，统一处理框线、短标签、图例栏与标签避让；分离式端到端与统一多任务端到端都改为调用该模块生成预览图。

**Tech Stack:** Python 3.11, OpenCV, NumPy, pytest

---

### Task 1: 新增共享渲染模块

**Files:**
- Create: `src/dltr/visualization/end_to_end_rendering.py`
- Test: `tests/visualization/test_end_to_end_rendering.py`

**Step 1: 写失败测试**
- 覆盖多框标签、长文本截断、图例区输出、画布尺寸扩展

**Step 2: 运行测试确认失败**
- `uv run pytest tests/visualization/test_end_to_end_rendering.py -q`

**Step 3: 实现最小共享渲染器**
- 渲染框线
- 渲染短标签
- 右侧图例栏
- 简单标签避让

**Step 4: 运行测试确认通过**
- `uv run pytest tests/visualization/test_end_to_end_rendering.py -q`

### Task 2: 接入分离式端到端渲染

**Files:**
- Modify: `src/dltr/pipeline/end_to_end.py`
- Test: `tests/pipeline/test_end_to_end.py`

**Step 1: 写失败测试**
- 验证预览图仍能生成，并通过共享渲染器产出

**Step 2: 运行测试确认失败**
- `uv run pytest tests/pipeline/test_end_to_end.py -q`

**Step 3: 接入共享渲染器**
- 删除本地重复绘制逻辑
- 改为统一渲染器生成 preview

**Step 4: 运行测试确认通过**
- `uv run pytest tests/pipeline/test_end_to_end.py -q`

### Task 3: 接入统一多任务端到端渲染

**Files:**
- Modify: `src/dltr/models/end2end_system.py`
- Test: `tests/pipeline/test_end_to_end.py`

**Step 1: 写失败测试**
- 验证 unified session 也走同一套 preview 渲染

**Step 2: 运行测试确认失败**
- `uv run pytest tests/pipeline/test_end_to_end.py -q`

**Step 3: 接入共享渲染器**
- 删除复制的 `_draw_polygon`
- 改为统一渲染器

**Step 4: 运行测试确认通过**
- `uv run pytest tests/pipeline/test_end_to_end.py -q`

### Task 4: 回归验证与文档

**Files:**
- Modify: `change_records/2026-04-09-*.md`

**Step 1: 全量验证**
- `uv run pytest -q`
- `uv run ruff check .`

**Step 2: 记录变更**
- 增加 change record，记录本次统一渲染改动和验证结果

