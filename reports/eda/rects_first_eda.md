# Chinese Scene-Text Dataset EDA Summary

## Validation Status

- Overall status: ACTION_REQUIRED
- Required datasets missing: 1
- Invalid dataset locations: 0

## Dataset Matrix

| Dataset | Required | Exists | Under data/raw | Images | Labels | Label Ratio |
|---|---:|---:|---:|---:|---:|---:|
| rctw17 | False | False | True | 0 | 0 | 0.000 |
| rects | True | True | True | 20000 | 40000 | 1.000 |
| shopsign | False | False | True | 0 | 0 | 0.000 |
| ctw | False | False | True | 0 | 0 | 0.000 |
| mtwi | False | False | True | 0 | 0 | 0.000 |
| ctr_benchmark_scene_lmdb | True | False | True | 0 | 0 | 0.000 |
| text_renderer_corpus | False | False | True | 0 | 0 | 0.000 |

## Hard-Case Heuristic Snapshot

### ctr_benchmark_scene_lmdb

- Total images scanned: 0
- Deep path ratio: 0.000
- Long filename ratio: 0.000
- Top keyword groups: small_text_like=0.000, dense_text_like=0.000, rotated_text_like=0.000
- Recommendation: No images found. Verify dataset paths before EDA.

### ctw

- Total images scanned: 0
- Deep path ratio: 0.000
- Long filename ratio: 0.000
- Top keyword groups: small_text_like=0.000, dense_text_like=0.000, rotated_text_like=0.000
- Recommendation: No images found. Verify dataset paths before EDA.

### mtwi

- Total images scanned: 0
- Deep path ratio: 0.000
- Long filename ratio: 0.000
- Top keyword groups: small_text_like=0.000, dense_text_like=0.000, rotated_text_like=0.000
- Recommendation: No images found. Verify dataset paths before EDA.

### rctw17

- Total images scanned: 0
- Deep path ratio: 0.000
- Long filename ratio: 0.000
- Top keyword groups: small_text_like=0.000, dense_text_like=0.000, rotated_text_like=0.000
- Recommendation: No images found. Verify dataset paths before EDA.

### rects

- Total images scanned: 20000
- Deep path ratio: 0.000
- Long filename ratio: 0.000
- Top keyword groups: small_text_like=0.000, dense_text_like=0.000, rotated_text_like=0.000
- Recommendation: Small-text-like samples appear low. Consider targeted augmentation.
- Recommendation: Rotated-text-like samples appear low. Add affine/rotation augmentation.
- Recommendation: Low-quality-like samples appear low. Add blur/noise simulation.

### shopsign

- Total images scanned: 0
- Deep path ratio: 0.000
- Long filename ratio: 0.000
- Top keyword groups: small_text_like=0.000, dense_text_like=0.000, rotated_text_like=0.000
- Recommendation: No images found. Verify dataset paths before EDA.

### text_renderer_corpus

- Total images scanned: 0
- Deep path ratio: 0.000
- Long filename ratio: 0.000
- Top keyword groups: small_text_like=0.000, dense_text_like=0.000, rotated_text_like=0.000
- Recommendation: No images found. Verify dataset paths before EDA.

## Next Steps

- Fill missing required dataset paths before training.
- Regenerate EDA after each new dataset import.
- Use hard-case ratios to define targeted augmentation policies.
