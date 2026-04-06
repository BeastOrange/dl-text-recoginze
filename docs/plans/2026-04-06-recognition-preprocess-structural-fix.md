# Recognition Preprocess Structural Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修正识别链路对竖排/高纵横比场景文本的不适配问题，让训练预处理和推理预处理完全同构。

**Architecture:** 新增共享识别预处理模块，统一负责灰度化、竖排方向归一化、保持纵横比缩放和 padding。训练集读取、单图推理、批量推理全部复用这一模块，并把相关参数固化到识别配置与 checkpoint 中。

**Tech Stack:** Python, PyTorch, OpenCV, pytest

---

### Task 1: 写失败测试锁定预处理行为

**Files:**
- Create: `tests/recognition/test_preprocessing.py`
- Modify: `tests/recognition/test_config.py`

**Steps:**
1. 写一个测试，要求竖排文本图像在启用方向归一化时会被旋转为横向布局。
2. 写一个测试，要求保持纵横比预处理时内容不会被直接拉伸到满宽，而是右侧保留 padding。
3. 写一个配置测试，要求识别配置支持预处理参数并能保留默认值。
4. 运行：`uv run pytest tests/recognition/test_preprocessing.py tests/recognition/test_config.py -q`
5. 预期：失败，提示缺少预处理模块或配置字段。

### Task 2: 实现共享识别预处理

**Files:**
- Create: `src/dltr/models/recognition/preprocessing.py`
- Modify: `src/dltr/models/recognition/config.py`
- Modify: `src/dltr/models/recognition/__init__.py`

**Steps:**
1. 新增预处理配置 dataclass，包含是否保持纵横比、是否做竖排归一化、竖排阈值、padding 值。
2. 实现共享预处理函数，输出固定尺寸灰度数组和元信息。
3. 在识别配置中解析/校验预处理参数，并保持旧配置兼容。
4. 运行：`uv run pytest tests/recognition/test_preprocessing.py tests/recognition/test_config.py -q`
5. 预期：新增测试通过。

### Task 3: 接入训练与推理链路

**Files:**
- Modify: `src/dltr/models/recognition/trainer.py`
- Modify: `src/dltr/models/recognition/inference.py`
- Modify: `configs/recognition/transformer_4090.yaml`
- Modify: `configs/recognition/transformer_detector_crop_4090.yaml`

**Steps:**
1. 让训练数据集和推理会话都使用共享预处理模块，移除各自的直接 resize 逻辑。
2. 让 checkpoint 保存并恢复预处理配置。
3. 在 4090 识别配置中显式声明预处理策略，便于服务器复现实验。
4. 运行：`uv run pytest tests/recognition/test_trainer.py tests/pipeline/test_end_to_end.py -q`
5. 预期：训练与推理相关测试通过。

### Task 4: 收尾验证

**Files:**
- Modify: `change_records/` 下对应记录（如需要）

**Steps:**
1. 运行：`uv run ruff check .`
2. 运行：`uv run pytest -q`
3. 记录本次结构性改动和下一轮训练命令。
4. 准备提交：`git status --short`
