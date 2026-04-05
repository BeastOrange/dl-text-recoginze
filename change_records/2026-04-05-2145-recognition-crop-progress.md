# 2026-04-05 21:45 识别裁剪进度反馈

## 变更内容

- 为 `src/dltr/data/recognition_crops.py` 增加实时进度条

## 目的

- 避免 `data prepare-recognition-crops` 在服务器前台运行时看起来像“卡住”
- 让用户能实时看到当前 split 的处理进度、已生成裁剪数和跳过数

## 效果

- 执行 `data prepare-recognition-crops` 时会显示 `train/val/test` 的裁剪进度
- 终端会持续输出当前裁剪数量与跳过数量
