# 基于深度学习的中文自然场景文本检测与识别系统毕业设计实施方案

## 1. 项目定位

本项目的**主线任务**只有两条：

1. 中文自然场景文本检测
2. 中文自然场景文本识别

项目的**方法定位**是深度学习，不是语义任务项目。  
因此，毕业设计的核心口径统一为：

- 这是一个基于深度学习的 OCR 项目
- 主体工作围绕检测与识别展开
- OCR 后规则理解仅作为扩展章节，用于回应老师提出的“加入理解能力”的要求

## 2. 项目目标

本项目要完成一个完整、可训练、可演示、可答辩的中文自然场景文本系统，覆盖：

1. 中文文本区域检测
2. 中文文本识别
3. 端到端推理与可视化展示
4. Mac 开发、Linux GPU 训练、Windows 推理交付
5. 英文图表与实验报告输出

项目需要正面回应老师的三项要求：

- 不能只做普通 OCR，必须实现自然场景文本识别
- 工作量必须充足，不能停留在简单模型调用
- 需要体现一定的“识别后理解”能力，但不再把语义任务写成并列主线

## 3. 总体技术路线

### 3.1 研究主线

- 开发语言：Python
- 包管理器：uv
- 深度学习框架：PyTorch
- 本地开发：Mac
- 训练环境：Linux + RTX 4090 / 5090 级 GPU
- 客户交付：Windows，支持 CPU 与 CUDA 推理

### 3.2 系统结构

系统按四层主线展开：

1. 数据层：数据校验、格式转换、EDA、hard-case 分析
2. 检测层：定位自然场景中的中文文本区域
3. 识别层：对文本区域做裁剪、矫正与识别
4. 展示层：端到端推理、英文报告、中文演示界面

另设一个**扩展层**：

5. OCR 后规则理解扩展：关键词、时间、电话、价格、警示词等结构化抽取

说明：

- 扩展层不是项目主任务
- 扩展层不参与主实验主榜排名
- 扩展层主要用于答辩展示“识别后理解能力”

## 4. 模型方案

### 4.1 主线模型

- 检测主线：DBNet
- 识别主线：CRNN-CTC

选择依据：

- 可真实训练
- 可稳定复现
- 与当前仓库代码和数据处理链路一致
- 能支持后续创新点实验与答辩展示

### 4.2 降级为备选或展望的模型

以下内容不再写入主线验收方案：

- TransOCR 不再作为主识别模型
- MacBERT 不再作为核心训练模块
- 独立语义分类模型不再作为主实验主线

这些内容只保留为：

- 预研路线
- 备选增强
- 展望章节

## 5. 数据集方案

### 5.1 主检测数据集

优先使用以下公开中文自然场景数据：

- ReCTS：主检测数据集
- ShopSign：主补充数据集

可选增强：

- MTWI 2018
- RCTW-17
- CTW

说明：

- `ReCTS + ShopSign` 足以支撑本项目主线实验
- 其他数据集属于增强项，不作为主线阻塞条件

### 5.2 主识别数据来源

识别数据采用两类来源：

1. 由检测标注裁剪得到的真实场景文本行样本
2. 已有中文识别公开数据或少量合成样本作为补充

当前主线识别数据优先依赖：

- 从 ReCTS 标注中裁剪得到的识别样本
- 从 ShopSign 标注中裁剪得到的识别样本

### 5.3 为什么不把 OCR 数据集直接当作语义监督数据集

当前 OCR 数据集已经标注好，但它们的标注内容主要是：

- 文本框坐标
- 文本转写内容

这些标注天然适用于：

- 检测训练
- 识别训练

但它们并不天然提供：

- 语义类别标签
- 实体边界
- 结构化键值关系

因此，本项目不再把“语义监督训练”写成主线目标，而是把 OCR 后理解能力改成**规则扩展能力**。

### 5.4 数据目录约定

所有手动下载的数据物理整理到 `data/raw/` 下。

默认目录：

- `data/raw/rctw17/`
- `data/raw/rects/`
- `data/raw/shopsign/`
- `data/raw/ctw/`
- `data/raw/mtwi/`
- `data/raw/ctr_benchmark_scene_lmdb/`
- `data/raw/text_renderer_corpus/`
- `data/interim/`
- `data/processed/`

说明：

- 识别后规则分析不再依赖独立训练数据目录
- 所有扩展仅基于 OCR 结果做规则抽取与展示

## 6. 创新点设计

### 6.1 创新点一：中文 hard-case 感知训练与评估

围绕以下中文自然场景难例建立专项分析与实验：

- 小字
- 密集字
- 倾斜字
- 竖排字
- 遮挡字
- 低清晰度字
- 强反光字
- 艺术字体

实现方式：

- hard-case 样本统计
- hard-case 样本画廊
- hard-case 过采样
- 多尺度增强
- hard-case 单独评估报表

### 6.2 创新点二：选择性二次识别增强

系统先完成第一遍识别，再根据低置信度样本触发第二遍识别增强。

二次识别前处理固定为：

- 几何矫正
- 对比度增强
- 轻量锐化

实验比较三组：

- 不做二次识别
- 全量样本都做二次识别
- 仅低置信度样本做二次识别

### 6.3 创新点三：OCR 后规则理解扩展

本项目不把语义分类写成主线训练任务，而是将“理解能力”落实为 OCR 后规则化扩展：

- 关键词提取
- 时间抽取
- 电话抽取
- 价格抽取
- 警示词抽取
- 场景提示信息抽取

输出字段统一为：

- `keywords`
- `phone`
- `price`
- `time`
- `location_hint`
- `warning_terms`

该模块仅作为扩展章节，用于展示“识别后理解”能力。

## 7. 实验设计

### 7.1 检测实验

- `Det-B0`: DBNet baseline
- `Det-B1`: DBNet + hard-case sampling
- `Det-B2`: DBNet + hard-case sampling + multi-scale augmentation

指标：

- Precision
- Recall
- Hmean

### 7.2 识别实验

- `Rec-B0`: CRNN baseline
- `Rec-B1`: CRNN + hard-case sample rebalance
- `Rec-B2`: CRNN + selective second-pass refinement

指标：

- Word Accuracy
- CER
- NED
- Mean Edit Distance

### 7.3 系统级实验

- `Sys-B0`: Detection + Recognition baseline
- `Sys-B1`: Detection + Recognition + hard-case enhancement
- `Sys-B2`: Detection + Recognition + second-pass refinement

指标：

- End-to-End line accuracy
- Latency
- FPS
- CPU / GPU comparison

### 7.4 扩展章节实验

该部分不进入主榜，只作为扩展展示：

- OCR 后规则抽取案例展示
- 典型成功案例
- 典型失败案例
- 识别错误如何影响结构化抽取结果

## 8. 可视化要求

所有图表默认使用英文标题、英文坐标轴、英文图例。

### 8.1 数据阶段

- dataset overview
- source distribution
- text length distribution
- polygon area distribution
- orientation distribution
- hard-case distribution
- sample gallery

### 8.2 数据处理阶段

- raw vs cleaned comparison
- crop quality gallery
- augmentation gallery

### 8.3 训练阶段

- train / val loss curves
- precision / recall / Hmean curves
- CER / NED curves
- confidence histogram

### 8.4 评估阶段

- hard-case comparison chart
- ablation chart
- latency chart
- qualitative success / failure gallery
- end-to-end pipeline examples

## 9. 工程结构与主线命令

```text
.
├── PLAN.md
├── pyproject.toml
├── src/dltr/
├── configs/
├── scripts/
├── reports/
├── artifacts/
├── tests/
├── change_records/
└── .github/workflows/
```

主线 CLI：

- `uv run python scripts/run_dltr.py data validate`
- `uv run python scripts/run_dltr.py data stats`
- `uv run python scripts/run_dltr.py data prepare-detection`
- `uv run python scripts/run_dltr.py data prepare-recognition-crops`
- `uv run python scripts/run_dltr.py train detector`
- `uv run python scripts/run_dltr.py train recognizer`
- `uv run python scripts/run_dltr.py evaluate detector`
- `uv run python scripts/run_dltr.py evaluate recognizer`
- `uv run python scripts/run_dltr.py evaluate end2end`
- `uv run python scripts/run_dltr.py export onnx`
- `uv run python scripts/run_dltr.py demo`
- `uv run python scripts/run_dltr.py sync linux`

扩展 CLI：

- `uv run python scripts/run_dltr.py train end2end`
- `uv run python scripts/run_dltr.py evaluate end2end`

说明：

- 扩展 CLI 保留是为了兼容历史实验与附录
- 扩展 CLI 不再属于主线验收链路

## 10. 跨平台策略

### 10.1 Mac

- 日常开发
- 数据探索
- 报告与图表生成
- CPU / MPS smoke test

### 10.2 Linux + RTX GPU

- 正式训练
- 批量评估
- 权重导出与推理验证

代码同步方式：

- `rsync + ssh`

### 10.3 Windows

- 运行中文前端演示
- 执行 CPU / CUDA 推理
- 展示识别与结构化输出结果

Windows 默认不承担正式训练。

## 11. 过程治理

### 11.1 GitHub

- 所有可验证改动必须提交到 GitHub
- 每次改动都必须增加一条 `change_records/*.md`

### 11.2 提交信息

格式：

`<type>(scope): <中文动词开头摘要>`

示例：

- `feat(data): 建立中文场景数据分析流程`
- `feat(cli): 搭建项目命令行入口`
- `test(core): 补充路径校验单元测试`

## 12. 阶段拆分

### 阶段一：项目底座

- 初始化仓库
- 建立 uv 工程
- 搭建 CLI 骨架
- 建立 change_records 规范
- 建立 CI 与同步脚本

### 阶段二：数据治理与 EDA

- 数据目录校验
- 标注转换
- 数据统计分析
- hard-case 分析
- 英文图表生成

### 阶段三：检测主线

- DBNet baseline
- 检测训练与评估
- 检测可视化

### 阶段四：识别主线

- CRNN baseline
- 识别训练与评估
- 二次识别增强实验

### 阶段五：创新点增强

- hard-case 策略实验
- 二次识别策略实验
- 消融实验

### 阶段六：系统集成与交付

- 端到端 pipeline
- ONNX 导出
- Windows demo
- 最终英文图表与答辩材料

### 阶段七：扩展章节

- OCR 后规则理解案例
- 结构化字段展示
- 成功 / 失败案例分析

## 13. 答辩验收口径

答辩时的主结论只能来自以下部分：

- 检测实验
- 识别实验
- 系统级实验
- 工程化交付与展示

扩展章节的结论只能作为：

- 理解能力补充展示
- 失败案例分析
- 系统扩展能力说明

不能把扩展章节当作项目主榜核心结论。

## 14. 当前仓库实现策略

当前仓库从零构建，不接入客户已有代码。当前主线优先级为：

1. `PLAN.md` 总方案
2. uv 工程骨架
3. CLI 与公共路径工具
4. 数据校验、EDA 和报告生成
5. 检测与识别主线训练
6. Windows bootstrap 与 Linux rsync 脚本
7. CI、测试和 change_records

说明：

- 语义相关代码目前保留在仓库中，视为历史 baseline 与扩展模块
- 这些代码不再决定项目主线方向

## 15. 参考资料

- uv: https://docs.astral.sh/uv/
- PyTorch: https://docs.pytorch.org/get-started/locally/
- DBNet: https://github.com/MhLiao/DB
- ReCTS: https://arxiv.org/abs/1912.09641
- ShopSign: https://arxiv.org/abs/1903.10412
- RCTW-17: https://arxiv.org/abs/1708.09585
- CTW: https://cg.cs.tsinghua.edu.cn/people/~kun/2019ctw/ctw_jcst.pdf
- FudanVI Chinese Text Recognition Benchmark: https://github.com/FudanVI/benchmarking-chinese-text-recognition
- Benchmark paper: https://arxiv.org/abs/2112.15093
