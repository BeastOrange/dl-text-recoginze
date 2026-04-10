# 2026-04-10 English Benchmark Pipeline

- Added configurable recognition manifest formats for English OCR datasets:
  - `mjsynth`
  - `pairs`
  - `icdar_gt`
- Added English benchmark reporting flow:
  - recognition evaluation now writes JSON bundles
  - new `report summarize-english-benchmark` command computes `main/hard` averages
- Added English dataset and recognition config examples for MJSynth, IIIT5K, SVT, ICDAR13, and ICDAR15.
