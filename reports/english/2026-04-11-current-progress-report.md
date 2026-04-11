# 当前进度简报（英文 OCR Benchmark）

## 1. 当前状态

当前阶段已完成英文 OCR benchmark 评测流程，并得到可展示的阶段性结果。  
本轮仅汇报当前英文 benchmark 方案的结果，不再展开此前自训练尝试过程。

当前主结果基于两个英文场景文字数据集：

- `IIIT5K`
- `SVT`

评测结果已生成：

- 汇总图：`english_benchmark_summary.png`
- 汇总说明：`english_benchmark_summary.md`
- 单数据集结果：
  - `paddleocr_iiit5k_alnum_recognition_eval.md`
  - `paddleocr_svt_alnum_recognition_eval.md`

## 2. 当前核心结果

| 数据集 | 样本数 | Word Accuracy | CER | NED |
|---|---:|---:|---:|---:|
| IIIT5K | 3000 | 0.9480 | 0.0148 | 0.0173 |
| SVT | 257 | 0.8949 | 0.0326 | 0.0318 |

主结果平均值：

- `Main-English-Accuracy = 0.921471`

说明：

- 当前主表采用统一的英文文本规范化评测口径；
- 该口径下，`IIIT5K` 结果接近 `95%`，`SVT` 结果接近 `90%`；
- 目前已经能够支撑阶段性展示与答辩材料编写。

## 3. 可视化产物

### 3.1 Benchmark 结果图

- `english_benchmark_summary.png`

图中展示：

- `IIIT5K` 的识别准确率
- `SVT` 的识别准确率
- 英文主结果平均值

### 3.2 训练过程图

虽然当前最终采用的是 benchmark 评测方案，但训练阶段的历史记录也已经整理出来，可作为“尝试过程”的辅助材料：

- `training/transformer_english_iiit_svt_train_training_curve.png`
- `training/crnn_english_iiit_svt_train_training_curve.png`
- `training/english_training_comparison.png`

### 3.3 数据集探索图

当前英文数据集探索结果也已补齐：

- `../eda/english_benchmark_dataset_overview.png`
- `../eda/english_benchmark_text_length_distribution.png`
- `../eda/english_benchmark_dataset_samples.png`

这些图展示了：

- 数据集样本规模对比
- 文本长度分布
- 典型样本画廊

## 4. 当前阶段结论

当前英文 OCR 主线已经形成一套完整的可展示结果：

1. 已具备英文 benchmark 数据集评测能力；
2. 已得到可直接写入阶段性汇报的指标结果；
3. 已补齐 benchmark 图表、训练过程图和数据探索图；
4. 当前结果能够支持后续论文撰写与阶段汇报展示。

## 5. 下一步

下一步工作建议聚焦两项：

1. 将当前英文 benchmark 结果整理为论文中的“实验结果”章节；
2. 继续补充若干案例级可视化，增强展示效果。
