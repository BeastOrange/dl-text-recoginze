# 2026-04-15 English Two-Stage Fallback Without MJSynth

- Added fallback warmup config for environments without MJSynth raw data:
  - `configs/recognition/transformer_english_warmup_iiit.yaml`
- Purpose:
  - Stage A warmup on `iiit5k_train` only.
  - Stage B finetune on `iiit5k_train + svt_train` by resuming from Stage A checkpoint.
- Kept recognition diagnostics and OOV guardrail enabled in warmup config:
  - `max_oov_ratio`
  - `diagnostics_top_k`
