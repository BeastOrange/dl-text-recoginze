# Detector Runtime Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 detector 训练补充 CUDA 场景下的 AMP 与 DataLoader 运行时优化，提升 4090 训练吞吐。

**Architecture:** 保持 detector 训练逻辑不变，只在 runtime 层引入与 recognizer 对齐的优化能力：`pin_memory`、`persistent_workers`、`prefetch_factor`、`non_blocking`、`autocast` 与 `GradScaler`。

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: 为 detector 增加运行时配置对象与辅助函数

**Files:**
- Modify: `src/dltr/models/detection/trainer.py`
- Test: `tests/detection/test_trainer.py`

### Task 2: 在 detector DataLoader 和训练循环中接入优化

**Files:**
- Modify: `src/dltr/models/detection/trainer.py`
- Test: `tests/detection/test_trainer.py`

### Task 3: 验证回归

**Files:**
- Modify: `change_records/2026-04-07-*.md`
- Test: `tests/detection/test_trainer.py`
