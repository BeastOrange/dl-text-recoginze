# 2026-04-06 13:25 题目主线对齐与端到端训练补齐

## Summary

- 移除错误的语义训练主线与相关配置、测试、汇总逻辑
- 新增 `post_ocr` 规则分析模块，保留 OCR 后结构化提取与简单分析展示
- 接入可训练的 Transformer-CTC 识别模型并切换为默认识别配置
- 实现真实 second-pass 图像重识别与 `train end2end` 系统级训练编排
- 美化 TTY 场景下的 `tqdm` 进度条，并将 `AGENTS.md` 加入 `.gitignore`

## Why

- 当前仓库中的语义模型训练主线与毕业设计题目和任务书口径冲突
- 题目要求主线聚焦自然场景文本检测、识别、端到端系统与识别后初步分析
- 识别侧需要从占位的 `transocr` 配置切换到真实可训练的 Transformer 实现
- 系统级实验入口和真实 second-pass 执行路径缺失，会影响训练与答辩口径

## Files Changed

- `src/dltr/cli.py`
- `src/dltr/commands.py`
- `src/dltr/models/recognition/config.py`
- `src/dltr/models/recognition/inference.py`
- `src/dltr/models/recognition/trainer.py`
- `src/dltr/pipeline/end_to_end.py`
- `src/dltr/post_ocr/`
- `configs/recognition/transformer_baseline.yaml`
- `tests/`
- `.gitignore`
- `PLAN.md`

## Verification

- `uv run ruff check .`
- `uv run pytest -q`
- `uv run python scripts/check_change_records.py`

## Next

- 在服务器上继续执行正式训练与系统级评估
- 根据训练结果补齐 4090 定制版 Transformer 配置与实验报告
