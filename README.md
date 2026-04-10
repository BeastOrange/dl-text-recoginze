# dl-text-recoginze

基于深度学习的中文自然场景文本检测与识别项目（毕业设计工程化实现）。

项目主线能力：
- 文本检测（Detection）
- 文本识别（Recognition）
- 端到端推理与可视化（End-to-End + Visualization）

## 项目目录（根目录）

- `src/dltr/`：核心 Python 包代码（CLI、数据、模型、推理、可视化）
- `configs/`：数据、检测、识别、语义相关配置文件
- `scripts/`：启动脚本、同步脚本、校验脚本
- `tests/`：单元测试与命令级测试
- `data/`：数据目录（`raw/`、`interim/`、`processed/`）
- `artifacts/`：训练产物与模型相关输出
- `reports/`：评估与可视化报告输出
- `docs/`：设计与计划文档
- `change_records/`：变更记录（CI 会检查）
- `PLAN.md`：项目总体方案说明

## 环境与启动

推荐使用 `uv`。

```bash
# 1) 安装依赖（开发环境）
uv sync --extra dev

# 2) 查看 CLI 帮助
uv run python scripts/run_dltr.py --help
```

说明：仓库统一使用 `scripts/run_dltr.py` 作为命令入口。

## 命令与作用

### 顶层命令

| 命令 | 作用 |
|---|---|
| `data` | 数据校验、统计、清单与划分生成 |
| `train` | 训练检测、识别、端到端模型 |
| `evaluate` | 评估检测/识别/端到端结果 |
| `report` | 汇总训练报告、硬例报告、消融模板与总览 |
| `export` | 导出模型（目前支持 ONNX） |
| `demo` | 生成 demo 资产或启动 Streamlit 演示 |
| `sync` | 同步工程到 Linux 训练服务器 |

---

### `data` 子命令

```bash
uv run python scripts/run_dltr.py data --help
```

| 子命令 | 作用 |
|---|---|
| `validate` | 校验数据集路径与结构是否可用 |
| `stats` | 生成数据统计报告 |
| `build-rec-lmdb` | 生成识别 LMDB 数据清单 |
| `prepare-recognition` | 生成识别训练所需清单、字符集与划分 |
| `prepare-recognition-crops` | 从检测结果裁剪识别样本并生成识别数据 |
| `prepare-detection` | 生成检测训练所需清单与划分 |

---

### `train` 子命令

```bash
uv run python scripts/run_dltr.py train --help
```

| 子命令 | 作用 |
|---|---|
| `detector` | 训练文本检测模型 |
| `recognizer` | 训练文本识别模型 |
| `end2end` | 训练端到端联合模型 |

---

### `evaluate` 子命令

```bash
uv run python scripts/run_dltr.py evaluate --help
```

| 子命令 | 作用 |
|---|---|
| `detector` | 评估检测指标（precision/recall/hmean） |
| `recognizer` | 评估识别指标（word accuracy/CER/NED 等） |
| `end2end` | 端到端评估（支持 `--image` 与 `--manifest`） |

---

### `report` 子命令

```bash
uv run python scripts/run_dltr.py report --help
```

| 子命令 | 作用 |
|---|---|
| `summarize-training` | 汇总指定 run 的主指标并生成图表/报告 |
| `summarize-project` | 汇总检测+识别项目级报告 |
| `build-index` | 生成训练报告索引页 |
| `build-all` | 一键构建训练相关报告（推荐） |
| `build-ablation-template` | 生成消融实验模板 |
| `build-hardcase` | 生成 hard-case 报告 |
| `build-ablation-overview` | 生成主线任务消融总览图 |

---

### `export` / `demo` / `sync`

| 命令 | 作用 |
|---|---|
| `export onnx` | 导出 ONNX 模型 |
| `demo` | 生成 demo 文本分析结果；加 `--serve` 启动 Streamlit |
| `sync linux` | 通过脚本同步代码到 Linux 服务器 |

## 常用启动示例

```bash
# 1) 数据校验
uv run python scripts/run_dltr.py data validate --config configs/data/datasets.example.yaml

# 2) 一键生成训练报告
uv run python scripts/run_dltr.py report build-all --output-dir reports/train

# 3) 端到端单图推理并输出可视化
uv run python scripts/run_dltr.py evaluate end2end \
  --image data/raw/shopsign/images/image_15.jpg \
  --end2end-run-dir artifacts/end2end/4090-mainline \
  --output-dir reports/eval
```

## 测试与代码检查

```bash
uv run pytest
uv run ruff check .
```

## 备注

- 若你修改了 `src/`、`tests/`、`configs/`、`scripts/`、`docs/`、`PLAN.md` 等目录，请同步新增 `change_records/` 记录，满足 CI 规范。
