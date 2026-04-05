# Summary

Added reproducible smoke-run configs for detection, recognition, and semantic training to support report generation.

# Why

The project needs a fast, repeatable way to generate real report artifacts without launching long full-dataset experiments every time.

# Files Changed

- `configs/detection/dbnet_report_smoke.yaml`
- `configs/recognition/crnn_report_smoke.yaml`
- `configs/semantic/char_linear_report_smoke.yaml`

# Verification

- These configs are consumed by the report-generation workflow commands and smoke training runs.

# Next

- Generate real `reports/train/`, `reports/hardcase/`, and `reports/ablation/` outputs using the smoke configs.
