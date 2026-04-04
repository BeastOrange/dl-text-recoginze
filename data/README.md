# Data Layout

This project keeps all runnable dataset entrypoints under `data/raw/`.

## Expected Structure

- `data/raw/rects/train/`
- `data/raw/rects/test_part1/`
- `data/raw/rects/test_part2/`
- `data/raw/shopsign/images/`
- `data/raw/shopsign/annotation/`
- `data/raw/ctr_benchmark_scene_lmdb/`
- `data/raw/rctw17/`
- `data/raw/ctw/`
- `data/raw/mtwi/`
- `data/raw/text_renderer_corpus/`

## Physical Organization

After manual downloads finish, move them into the standard layout physically:

```bash
mkdir -p data/raw/rects data/raw/shopsign/images
mv ReCTS-Train data/raw/rects/train
mv ReCTS_test_part1 data/raw/rects/test_part1
mv ReCTS_test_part2 data/raw/rects/test_part2
mv ShopSign_1265/annotation data/raw/shopsign/annotation
find ShopSign_1265 -maxdepth 1 -type f -exec mv {} data/raw/shopsign/images/ \;
rmdir ShopSign_1265
```

## Notes

- `ReCTS` training data lives under `train/img`, `train/gt`, and `train/gt_unicode`.
- `ShopSign` images live under `images/`, and labels live under `annotation/`.
- Current CLI commands support these two layouts directly.
- Preferred entrypoint: `uv run python scripts/run_dltr.py ...`
