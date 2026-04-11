# 2026-04-10 English Benchmark Pipeline

- Added configurable recognition manifest formats for English OCR datasets:
  - `mjsynth`
  - `pairs`
  - `icdar_gt`
  - `iiit5k_mat`
  - `svt_xml`
- Added English benchmark reporting flow:
  - recognition evaluation now writes JSON bundles
  - new `report summarize-english-benchmark` command computes `main/hard` averages
- Added pretrained benchmark evaluation flow:
  - new `evaluate recognizer-benchmark`
  - supports `easyocr` as an optional runtime backend via `uv run --with easyocr ...`
- Added English dataset and recognition config examples for MJSynth, IIIT5K, SVT, ICDAR13, and ICDAR15.
- Added a local English dataset config for the current workspace paths:
  - `data/IIIT5K`
  - `data/svt1`
