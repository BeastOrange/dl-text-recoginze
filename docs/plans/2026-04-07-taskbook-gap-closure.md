# Taskbook Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 补齐任务书主线缺口，使仓库具备统一多任务训练、统一推理、真实 ONNX 导出与非空主线实验结果。

**Architecture:** 在保留现有检测/识别独立链路的前提下，新增统一多任务模型和统一 checkpoint 推理；同时将识别主线切换到检测裁切样本，并放宽 polygon 几何支持。

**Tech Stack:** Python 3.11, PyTorch, OpenCV, pytest, YAML CLI

---

### Task 1: 落地统一多任务模型

**Files:**
- Create: `src/dltr/models/end2end_system.py`
- Modify: `src/dltr/commands.py`
- Modify: `src/dltr/cli.py`
- Test: `tests/test_end2end_training_command.py`

### Task 2: 扩展统一 checkpoint 推理

**Files:**
- Modify: `src/dltr/pipeline/end_to_end.py`
- Modify: `src/dltr/pipeline/end_to_end_baseline.py`
- Test: `tests/pipeline/test_end_to_end.py`
- Test: `tests/pipeline/test_end_to_end_baseline.py`

### Task 3: 泛化 polygon 支持

**Files:**
- Modify: `src/dltr/data/detection_preparation.py`
- Modify: `src/dltr/models/detection/dataset.py`
- Modify: `src/dltr/data/recognition_crops.py`
- Modify: `src/dltr/pipeline/end_to_end.py`
- Modify: `src/dltr/pipeline/end_to_end_baseline.py`
- Test: `tests/data/test_detection_preparation.py`
- Test: `tests/data/test_recognition_crops.py`

### Task 4: 实现真实 ONNX 导出

**Files:**
- Modify: `src/dltr/cli.py`
- Modify: `src/dltr/commands.py`
- Create: `src/dltr/models/detection/export.py`
- Test: `tests/test_commands.py`

### Task 5: 补主线数据与真实报告

**Files:**
- Modify: `src/dltr/commands.py`
- Modify: `src/dltr/visualization/training_reports.py`
- Add: `change_records/2026-04-07-*.md`

