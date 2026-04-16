# SynthText / MJSynth 清单标签解析对齐

## 变更

- `src/dltr/data/manifest.py`：`mjsynth` 格式的 `_extract_mjsynth_text` 改为调用 `parse_mjsynth_label_from_stem`，与 `english_recognition_sources` 一致。
  - 修复此前仅用 `parts[1]` 导致的错误：SynthText 风格文件名 `ant+hill_100_0` 被标成 `100` 而非 `ant+hill`。
- `tests/data/test_manifest_and_reporting.py`：增加 SynthText 风格词片文件名的回归用例。
- `configs/data/datasets.synthtext.autodl.yaml`：AutoDL 上 `data/SynthText/SynthText` 布局的数据集声明。
- `configs/recognition/transformer_english_synthtext.yaml`：SynthText 预训练示例配置（路径与 README 英文 benchmark 一致）。

## 验证

```bash
uv run pytest tests/data/test_manifest_and_reporting.py -q
uv run ruff check src/dltr/data/manifest.py
```
