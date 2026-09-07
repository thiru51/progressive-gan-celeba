# progressive-gan-celeba

ProGAN introduces four things at once — equalised learning rates, pixel norm,
minibatch stddev and progressive growing. This repo turns each on separately,
on the same network and the same budget, to see what each one is actually worth.

CelebA at 64x64, 12,000 steps, batch 64, 3 seeds per arm.

## Results

FID, mean over 3 seeds (lower is better). Comparable within this repo only —
the features come from torchvision's InceptionV3, not the TF-Slim graph used in
published FID tables.

| arm | FID | vs baseline |
|---|---|---|
| dcgan (baseline) | 31.51 | — |
| eqlr | 293.30 | +261.79 |
| eqlr-matched | 23.73 | -7.78 |
| pixelnorm | 31.15 | -0.36 |
| mbstd | 31.71 | +0.20 |
| dcgan-all | 22.70 | -8.80 |
| progan-fixed | 28.38 | -3.13 |
| progan-grow | 54.31 | +22.80 |

Two runs of the same arm at the same seed differ by about 2 FID, so anything
smaller than that is noise.

- The equalised learning rate does essentially all of the work.
- Pixel norm and minibatch stddev do nothing measurable on their own.
- Progressive growing is worse, not better, at this resolution and budget.
  Repeating it at 128x128 widens the gap rather than closing it.

Full write-up, including how the equalised learning rate has to be corrected
before it works at all, is in [RESULTS.md](RESULTS.md).

## Running it

```bash
pixi install
pixi run python scripts/fetch_celeba.py
pixi run python scripts/run_ablation.py --seeds 0 1 2
```

Writes `artifacts/ablation.json` with the per-seed numbers behind the table.

```bash
pixi run pytest -q
```

## Layout

```
src/pgan/      models, data, training, FID
scripts/       data fetch, ablation runner, reporting
tests/         80 tests
artifacts/     per-seed results as JSON
```
