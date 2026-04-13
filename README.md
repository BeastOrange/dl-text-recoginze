# dl-text-recoginze

基于深度学习的中文场景文本检测与识别项目，包含检测、识别、端到端推理、报告汇总与
OCR 后处理分析能力。

## 项目主线

- 文本检测：DBNet 风格训练、评估摘要、导出
- 文本识别：CRNN / Transformer 训练、评估摘要、benchmark 汇总
- 端到端：检测 + 识别推理、manifest 批量评估、扫参
- 报告：训练汇总、项目总览、hard-case、英文 benchmark 汇总
- Demo：命令行 demo 与 Streamlit 演示

## 目录结构

- `src/dltr/`：核心 Python 包代码
- `configs/`：数据、检测、识别相关配置
- `scripts/`：仓库内统一命令入口与辅助脚本
- `tests/`：单元测试与命令级回归测试
- `data/`：数据目录
  - `data/raw/`：推荐放原始下载数据
  - `data/interim/`：中间产物
  - `data/processed/`：manifest、划分、字符集等生成物
- `artifacts/`：训练权重、运行目录、导出文件
- `reports/`：评估报告、汇总图表、demo 资产
- `change_records/`：CI 要求的变更记录

## 一条最短主线

如果你只想先跑通中文主线，按这个顺序执行：

1. 创建环境并安装依赖
2. 准备 `data/raw/...` 下的数据集
3. 运行 `data prepare-detection`
4. 运行 `data prepare-recognition`
5. 启动 `train detector`
6. 启动 `train recognizer`
7. 用 `evaluate end2end --image ...` 做单图推理
8. 用 `report build-all` 汇总训练报告

下面是完整操作手册。

## 1. 环境创建

推荐使用 `uv` 管理 Python 3.11 环境。

```bash
# 安装/准备 Python 3.11
uv python install 3.11

# 最小开发环境
uv sync --extra dev

# 如果要训练模型，建议安装训练依赖
uv sync --extra dev --extra train-cu

# 如果还需要 Streamlit demo 和可视化
uv sync --extra dev --extra train-cu --extra demo --extra viz
```

可选检查：

```bash
uv run python -V
uv run python scripts/run_dltr.py --help
```

说明：

- 仓库内最稳妥的命令入口是 `scripts/run_dltr.py`
- `uv run dltr ...` 依赖已安装的包入口；在未执行可编辑安装时不如脚本入口稳定
- `uv sync` 会自动维护项目 `.venv`
- IIIT5K 的 `.mat` 标注解析依赖 `scipy`，可直接使用
  `uv run --with scipy python scripts/run_dltr.py ...`
- EasyOCR benchmark 评测可直接使用
  `uv run --with easyocr python scripts/run_dltr.py ...`

## 2. 数据布局与配置

数据校验当前要求数据目录位于仓库的 `data/` 下；推荐把原始下载内容放在 `data/raw/`，
但也兼容当前仓库内已存在的本地 benchmark 布局，例如 `data/IIIT5K`、`data/svt1`。

常用配置：

| 配置文件 | 适用场景 |
|---|---|
| `configs/data/datasets.example.yaml` | 中文场景数据标准布局，默认放在 `data/raw/...` |
| `configs/data/datasets.english.example.yaml` | 英文 benchmark 标准布局，默认放在 `data/raw/...` |
| `configs/data/datasets.english.local.yaml` | 当前仓库已下载的本地 benchmark 布局：`data/IIIT5K`、`data/svt1` |

中文场景标准布局示例：

```text
data/raw/
├── rects/
├── shopsign/
├── rctw17/
├── ctw/
├── mtwi/
├── ctr_benchmark_scene_lmdb/
└── text_renderer_corpus/
```

英文 benchmark 标准布局示例：

```text
data/raw/
├── mjsynth/
├── iiit5k/
├── svt/
├── icdar13/
└── icdar15/
```

## 3. 数据集下载链接

中文场景 / 中文识别：

- RCTW-17: <http://rctw.vlrlab.net/>
- ReCTS（ICDAR 2019 Reading Chinese Text on Signboard）:
  <http://rrc.cvc.uab.es/?ch=12>
- ShopSign: <https://github.com/chongshengzhang/shopsign>
- CTW: <https://ctwdataset.github.io/>
- MTWI / ICPR MTWI 2018（天池入口）:
  <https://tianchi.aliyun.com/competition/entrance/231650/information>
- 中文识别 benchmark（FudanVI）:
  <https://github.com/FudanVI/benchmarking-chinese-text-recognition>

英文识别 / benchmark：

- MJSynth / Synthetic Word Dataset:
  <https://www.robots.ox.ac.uk/~vgg/data/text/>
- IIIT5K:
  <https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset>
- SVT:
  <https://tc11.cvc.uab.es/datasets/SVT_1>
- ICDAR 2013 Focused Scene Text:
  <https://rrc.cvc.uab.es/?ch=2>
- ICDAR 2015 Incidental Scene Text:
  <https://rrc.cvc.uab.es/?ch=4>

备注：

- 部分数据集需要注册、比赛权限或邮件申请后才能下载
- `ShopSign` 仓库对完整数据提供了申请与说明
- `IIIT5K` 的项目页和 TC11 索引页都可作为入口，这里优先放项目主页

## 4. 中文主线：数据处理

### 4.1 校验数据路径

```bash
uv run python scripts/run_dltr.py data validate \
  --config configs/data/datasets.example.yaml
```

作用：

- 检查数据目录是否位于仓库 `data/` 下
- 检查必需数据集是否存在
- 输出每个数据集的解析路径

### 4.2 生成数据 EDA 摘要

```bash
uv run python scripts/run_dltr.py data stats \
  --config configs/data/datasets.example.yaml
```

默认输出：

- `reports/eda/dataset_eda_summary.md` 或当前实现指定的 EDA Markdown 输出路径

### 4.3 生成检测训练数据

```bash
uv run python scripts/run_dltr.py data prepare-detection \
  --config configs/data/datasets.example.yaml \
  --datasets rects shopsign
```

这一步会生成：

- 检测 manifest：`data/processed/detection_manifests/*.jsonl`
- 合并 manifest：`data/processed/detection_combined.jsonl`
- 划分结果：`data/processed/detection_splits/train.jsonl`
- 划分结果：`data/processed/detection_splits/val.jsonl`
- 划分结果：`data/processed/detection_splits/test.jsonl`
- 摘要：`data/processed/detection_preparation_summary.md`

### 4.4 生成识别训练数据

直接从数据集标注生成识别训练输入：

```bash
uv run python scripts/run_dltr.py data prepare-recognition \
  --config configs/data/datasets.example.yaml \
  --datasets rects shopsign
```

这一步会生成：

- 单数据集 manifest：`data/processed/manifests/*.jsonl`
- 合并 manifest：`data/processed/recognition_combined.jsonl`
- 字符集：`data/processed/charset_zh_mixed.txt`
- 划分结果：`data/processed/recognition_splits/train.jsonl`
- 划分结果：`data/processed/recognition_splits/val.jsonl`
- 划分结果：`data/processed/recognition_splits/test.jsonl`
- 摘要：`data/processed/recognition_preparation_summary.md`

### 4.5 可选：从检测标注裁剪识别样本

如果你想让识别模型直接使用检测框裁剪后的样本，可以在检测划分生成后运行：

```bash
uv run python scripts/run_dltr.py data prepare-recognition-crops \
  --detection-split-dir data/processed/detection_splits \
  --crop-output-dir data/processed/recognition_crops \
  --recognition-split-dir data/processed/recognition_crop_splits
```

这一步会生成：

- 裁剪图目录：`data/processed/recognition_crops/`
- 裁剪版划分：`data/processed/recognition_crop_splits/`
- 合并 manifest：`data/processed/recognition_crop_combined.jsonl`
- 裁剪版字符集：`data/processed/recognition_crop_charset_zh_mixed.txt`
- 摘要：`data/processed/recognition_crop_preparation_summary.md`

## 5. 中文主线：训练启动

### 5.1 检测训练

```bash
uv run python scripts/run_dltr.py train detector \
  --config configs/detection/dbnet_baseline.yaml \
  --run-id det-rects-baseline
```

默认使用的输入：

- `configs/detection/dbnet_baseline.yaml`
- `data/processed/detection_splits/train.jsonl`
- `data/processed/detection_splits/val.jsonl`

默认输出根目录来自配置文件：

- `artifacts/detection/det_dbnet_baseline/<run_id>/`

该运行目录通常包含：

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `training_history.jsonl`
- `training_history.md`
- `training_curve.png`
- `training_summary.json`
- `reports/`

### 5.2 识别训练

```bash
uv run python scripts/run_dltr.py train recognizer \
  --config configs/recognition/transformer_baseline.yaml \
  --run-id rec-transformer-baseline
```

默认使用的输入：

- `data/processed/recognition_splits/train.jsonl`
- `data/processed/recognition_splits/val.jsonl`
- `data/processed/charset_zh_mixed.txt`

默认输出根目录来自配置文件：

- `artifacts/checkpoints/recognition/transformer_baseline/<run_id>/`

该运行目录通常包含：

- `best.pt`
- `last.pt`
- `training_history.jsonl`
- `training_history.md`
- `training_curve.png`
- `training_summary.json`
- 训练时生成的识别评估报告

如果你想切换到 CRNN，可改用 `configs/recognition/crnn_baseline.yaml`。

### 5.3 端到端联合训练

```bash
uv run python scripts/run_dltr.py train end2end \
  --detector-config configs/detection/dbnet_baseline.yaml \
  --recognizer-config configs/recognition/transformer_baseline.yaml \
  --run-id e2e-mainline
```

默认输出目录：

- `artifacts/end2end/e2e-mainline/`

该目录通常包含：

- `best.pt`
- `last.pt`
- `training_history.jsonl`
- `training_summary.json`
- `training_summary.md`

## 6. 评估、推理与报告

### 6.1 Detection 评估摘要归档

当前 `evaluate detector` 的语义是“把你已有的 precision / recall / hmean 写入报告”，
不是自动跑模型推理。

```bash
uv run python scripts/run_dltr.py evaluate detector \
  --config configs/detection/dbnet_baseline.yaml \
  --run-id det-rects-baseline \
  --split val \
  --precision 0.82 \
  --recall 0.79 \
  --hmean 0.80
```

默认输出到对应 detection run 目录下的：

- `reports/evaluation_val.md`
- `reports/evaluation_val.json`

### 6.2 Recognition 评估摘要归档

当前 `evaluate recognizer` 的语义同样是“把你已有的识别指标写入评估包”。

```bash
uv run python scripts/run_dltr.py evaluate recognizer \
  --run-name rec-transformer-baseline \
  --model-name transformer \
  --samples 3000 \
  --word-accuracy 0.91 \
  --cer 0.07 \
  --ned 0.08 \
  --mean-edit-distance 0.18 \
  --output-dir reports/eval
```

默认输出：

- `reports/eval/<run_name>_recognition_eval.md`
- `reports/eval/<run_name>_recognition_eval.json`

### 6.3 端到端单图推理

这是当前最接近“真实运行评估”的入口之一。请将图片路径替换成你自己的测试图片。

```bash
uv run python scripts/run_dltr.py evaluate end2end \
  --image path/to/your_image.jpg \
  --detector-run-dir artifacts/detection/det_dbnet_baseline/det-rects-baseline \
  --recognizer-run-dir artifacts/checkpoints/recognition/transformer_baseline/rec-transformer-baseline \
  --output-dir reports/eval
```

默认输出：

- `reports/eval/end_to_end_result.json`
- `reports/eval/end_to_end_result.md`
- `reports/eval/end_to_end_preview.png`

### 6.4 端到端批量 manifest 评估

```bash
uv run python scripts/run_dltr.py evaluate end2end \
  --manifest data/processed/detection_splits/val.jsonl \
  --detector-run-dir artifacts/detection/det_dbnet_baseline/det-rects-baseline \
  --recognizer-run-dir artifacts/checkpoints/recognition/transformer_baseline/rec-transformer-baseline \
  --output-dir reports/eval
```

默认输出：

- `reports/eval/end2end_baseline_summary.json`
- `reports/eval/end2end_baseline_summary.md`
- `reports/eval/end2end_error_analysis.json`
- `reports/eval/end2end_error_analysis.md`

### 6.5 端到端参数扫描

```bash
uv run python scripts/run_dltr.py evaluate end2end \
  --manifest data/processed/detection_splits/val.jsonl \
  --detector-run-dir artifacts/detection/det_dbnet_baseline/det-rects-baseline \
  --recognizer-run-dir artifacts/checkpoints/recognition/transformer_baseline/rec-transformer-baseline \
  --sweep \
  --sweep-detector-thresholds 0.3 0.4 0.5 \
  --sweep-min-areas 16 32 \
  --output-dir reports/eval
```

### 6.6 项目级训练报告

如果你已经有 detection 和 recognition 的多个 run，可以直接构建汇总：

```bash
uv run python scripts/run_dltr.py report build-all \
  --output-dir reports/train
```

默认会尝试从以下目录收集运行结果：

- `artifacts/detection/`
- `artifacts/checkpoints/recognition/`

通常会生成：

- `reports/train/detection_summary.*`
- `reports/train/recognition_summary.*`
- `reports/train/project_training_summary.*`
- `reports/train/index.md`
- `reports/ablation/ablation_overview.*`
- `reports/hardcase/hardcase_overview.*`

## 7. 英文 Benchmark 支线

标准布局示例（数据位于 `data/raw/...`）：

```bash
# 1) 生成英文识别训练/评测输入
uv run --with scipy python scripts/run_dltr.py data prepare-recognition \
  --config configs/data/datasets.english.example.yaml \
  --datasets mjsynth iiit5k svt icdar13 icdar15 \
  --split-output-dir data/processed/english_recognition_splits \
  --charset-output data/processed/charset_en_benchmark.txt

# 2) 训练英文识别模型
uv run python scripts/run_dltr.py train recognizer \
  --config configs/recognition/transformer_english_mjsynth.yaml

# 3) 写入单个 benchmark 结果（md + json）
uv run python scripts/run_dltr.py evaluate recognizer \
  --run-name transformer_iiit5k \
  --model-name transformer \
  --samples 3000 \
  --word-accuracy 0.91 \
  --cer 0.07 \
  --ned 0.08 \
  --mean-edit-distance 0.18 \
  --benchmark-name iiit5k \
  --benchmark-category main \
  --output-dir reports/eval

# 4) 汇总英文 benchmark
uv run python scripts/run_dltr.py report summarize-english-benchmark \
  --benchmark-jsons reports/eval/*_recognition_eval.json \
  --output-dir reports/english
```

当前仓库本地布局示例（数据位于 `data/IIIT5K`、`data/svt1`）：

```bash
uv run --with scipy python scripts/run_dltr.py data prepare-recognition \
  --config configs/data/datasets.english.local.yaml \
  --datasets iiit5k_test svt_train \
  --split-output-dir data/processed/english_recognition_splits \
  --charset-output data/processed/charset_en_benchmark.txt
```

预训练 benchmark 评测示例：

```bash
uv run --with easyocr python scripts/run_dltr.py evaluate recognizer-benchmark \
  --run-name easyocr_iiit5k \
  --model-name easyocr_en \
  --backend easyocr \
  --manifest data/processed/manifests/iiit5k.jsonl \
  --benchmark-name iiit5k \
  --benchmark-category main \
  --output-dir reports/eval
```

如果你使用的是本地布局配置，对应 manifest 文件名会是：

- `data/processed/manifests/iiit5k_test.jsonl`
- `data/processed/manifests/svt_train.jsonl`

## 8. 结果存放位置

常用结果目录如下：

| 目录 | 内容 |
|---|---|
| `data/processed/detection_manifests/` | detection 单数据集 manifest |
| `data/processed/detection_splits/` | detection 训练/验证/测试划分 |
| `data/processed/manifests/` | recognition 单数据集 manifest |
| `data/processed/recognition_splits/` | recognition 训练/验证/测试划分 |
| `data/processed/recognition_crops/` | 裁剪后的识别图片 |
| `artifacts/detection/<experiment>/<run_id>/` | detection 训练权重与摘要 |
| `artifacts/checkpoints/recognition/<experiment>/<run_id>/` | recognition 训练权重与摘要 |
| `artifacts/end2end/<run_id>/` | 端到端联合训练结果 |
| `artifacts/exports/` | 导出的 ONNX 文件 |
| `reports/eval/` | recognition / benchmark / end2end 评估输出 |
| `reports/train/` | detection / recognition / project 训练汇总 |
| `reports/english/` | 英文 benchmark 汇总 |
| `reports/hardcase/` | hard-case 图与 Markdown |
| `reports/ablation/` | 消融总览 |
| `reports/demo_assets/` | demo 生成资产 |

## 9. Demo 与辅助命令

命令行 demo：

```bash
uv run python scripts/run_dltr.py demo \
  --text "营业时间 09:00-21:00 电话 13800138000"
```

Streamlit：

```bash
uv run python scripts/run_dltr.py demo --serve
```

ONNX 导出：

```bash
uv run python scripts/run_dltr.py export onnx \
  --config configs/detection/dbnet_baseline.yaml \
  --checkpoint artifacts/detection/det_dbnet_baseline/det-rects-baseline/checkpoints/best.pt \
  --output artifacts/exports/model.onnx
```

## 10. 测试与检查

```bash
uv run pytest
uv run ruff check .
```

## 11. 变更记录要求

若你修改了 `src/`、`tests/`、`configs/`、`scripts/`、`docs/`、`README.md`、
`pyproject.toml`、`PLAN.md` 或其他受 CI 约束的路径，请同步新增一条
`change_records/` 记录。
