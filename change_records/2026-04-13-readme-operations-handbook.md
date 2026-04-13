# 2026-04-13 README Operations Handbook

- Expanded the root `README.md` from a quick overview into an operations handbook.
- Added detailed environment setup notes, including optional extras for training,
  demo, visualization, `scipy`, and benchmark backends.
- Added a full Chinese mainline workflow:
  - data validation
  - dataset EDA
  - detection manifest generation
  - recognition manifest generation
  - optional crop-based recognition preparation
  - detector / recognizer / end-to-end training
- Added evaluation and reporting guidance, including the current distinction
  between metric archival commands and true end-to-end inference commands.
- Added a dedicated artifact-location section to document where manifests,
  checkpoints, summaries, reports, and demo assets are stored.
