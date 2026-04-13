# 2026-04-13 README And Dataset Alignment

- Updated root `README.md` to match the current command surface and dataset layouts.
- Documented both standard `data/raw/...` layouts and the current local benchmark layout:
  - `data/IIIT5K`
  - `data/svt1`
- Fixed English benchmark examples so generated outputs align with the current
  recognition configs:
  - `data/processed/english_recognition_splits`
  - `data/processed/charset_en_benchmark.txt`
- Clarified that `data build-rec-lmdb` currently emits JSONL manifests.
- Added dataset source links for Chinese and English OCR datasets.
- Relaxed dataset path validation from `data/raw/` to the repository `data/` root,
  so local benchmark snapshots under `data/` validate correctly.
- Switched package metadata to use `README.md` as the project readme.
