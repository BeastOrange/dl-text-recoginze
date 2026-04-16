# 2026-04-15 CTC Blank Bias Guardrail

- Added `ctc_blank_bias` to recognition config (`RecognitionExperimentConfig`) to control
  classifier bias initialization for CTC blank class.
- Wired blank-bias application in trainer before optimization starts:
  - `src/dltr/models/recognition/trainer.py::_apply_ctc_blank_bias`
- Enabled this guardrail for English runs:
  - `configs/recognition/transformer_english_stable.yaml`
  - `configs/recognition/transformer_english_warmup_iiit.yaml`
- Added regression coverage:
  - config parsing test for `ctc_blank_bias`
  - unit test verifying only blank-class bias is adjusted
