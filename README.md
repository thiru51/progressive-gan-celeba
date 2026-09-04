# progressive-gan-celeba

ProGAN's paper introduces four changes to GAN training at once and reports the
combined result. This repository takes them apart and measures them one at a
time, against a DCGAN baseline, on CelebA at 64x64, under a step budget that
fits on one consumer GPU.

The question in one line: **which of ProGAN's four contributions is actually
load-bearing, and is progressive growing one of them?**

See [END_GOAL.md](END_GOAL.md) for the longer statement, [RESULTS.md](RESULTS.md)
for the measured numbers, and [HANDOFF.md](HANDOFF.md) for the state of the code.

---

## Status

Read this before reading anything else.

| Piece | State |
|---|---|
| Layers, models, growing schedule | Done. 74 tests pass. |
| Data pipeline | Done. 202,599 CelebA faces at 64x64, 20,000 held out for FID. |
| Training loop | Done. All seven arms train, checkpoint and reload. |
| FID | Done, and validated against closed-form cases. |
| The ablation sweep | See [RESULTS.md](RESULTS.md). |

**This is a small-scale attribution study, not a reproduction.** ProGAN trains to
1024x1024 over days on hardware that is not a laptop. Everything here is 64x64
under a 12,000-step budget. The point is not to match the paper's samples; it is
to find out which of its ingredients survives being isolated.

**FID here is not FID there.** The features come from torchvision's ImageNet
InceptionV3, not the TF-Slim graph every published FID table uses. The scale is
different. Numbers in this repository are comparable to each other and to
nothing else. This is stated again at the top of `src/pgan/metrics/fid.py` and
again in RESULTS.md, because it is the single easiest thing to misquote.

---

## Contents

- [What the four contributions actually do](#what-the-four-contributions-actually-do)
- [The seven arms](#the-seven-arms)
- [What is held fixed, and why it matters](#what-is-held-fixed-and-why-it-matters)
- [File-by-file layout](#file-by-file-layout)
- [Install](#install)
- [Running it, in order](#running-it-in-order)
- [Cost on a 12 GB GPU](#cost-on-a-12-gb-gpu)
- [Troubleshooting](#troubleshooting)
- [Honest limitations](#honest-limitations)
- [References](#references)

---

## What the four contributions actually do

### Equalised learning rate

The usual practice is to draw initial weights from a distribution scaled by
He's constant, `sqrt(2 / fan_in)`, and then leave them alone. ProGAN draws from
`N(0, 1)` and multiplies by that constant inside every forward pass instead.

Algebraically at initialisation these are the same network. They stop being the
same the moment Adam is involved. Adam divides each parameter's update by a
running estimate of its own gradient magnitude, so a layer whose weights are
small takes proportionally larger steps than a layer whose weights are large.
Baking the scale into the weights therefore hands different layers different
*effective* learning rates. Applying it after the optimiser has seen the weight
puts every layer on the same one.

`src/pgan/models/layers.py`, `EqualizedConv2d` / `EqualizedConvTranspose2d`.

> The transposed-convolution version is where this is easy to get wrong.
> `nn.ConvTranspose2d` stores its weight as `(in, out, kh, kw)` — the opposite
> layout from `nn.Conv2d` — so the fan-in is `out_channels * kh * kw`. Reading it
> off the first axis instead scales every generator layer by a constant, which
> presents as a learning rate that will not tune rather than as a visible bug.
> `test_equalized_transpose_uses_out_channels_as_fan_in` pins it.

### Pixelwise feature normalisation

After each generator convolution, normalise every pixel's feature vector across
channels to unit length. It has no learnable parameters, which is the point: it
cannot amplify anything, it can only prevent escalation. The failure mode it
targets is G and D entering an arms race on activation magnitude, where G answers
a stronger discriminator by making its features larger rather than by drawing
something different.

Here it *replaces* batch norm in the generator rather than joining it. Stacking
both normalises twice and the batch statistics stop carrying information.

### Minibatch standard deviation

One extra channel appended to the discriminator's last feature map, holding the
average standard deviation across the batch. This is the only place in the whole
architecture where the discriminator sees more than one sample at a time.

A generator that has collapsed to a single mode produces batches with almost no
spread. That lands as a near-constant feature map the discriminator can read off
directly, so collapse becomes trivially detectable and therefore expensive for G
to commit to. `test_minibatch_stddev_reads_zero_on_a_collapsed_batch` checks
exactly this property rather than just the output shape.

### Progressive growing

Train at 4x4 until stable, then fade in an 8x8 block, and so on to the target
resolution. The fade is a linear blend controlled by `alpha`: at `alpha = 0` the
new block contributes nothing and the network reproduces exactly what it produced
one resolution down, upsampled. Nothing already learned is discarded at the
moment a stage is added.

The argument for it is that the coarse structure of an image is settled at low
resolution, cheaply, before any capacity is spent on detail. The argument against
it, at 64x64, is that there is barely any coarse-to-fine range to exploit. That
is the disagreement this repository is set up to settle.

`src/pgan/schedule.py` and the `stage` / `alpha` arguments in
`src/pgan/models/progan.py`.

---

## The seven arms

Read top to bottom, this walks from DCGAN to ProGAN one change at a time.

| arm | architecture | eq. LR | pixel norm | minibatch std | growing |
|---|---|---|---|---|---|
| `dcgan` | DCGAN | | | | |
| `eqlr` | DCGAN | yes | | | |
| `pixelnorm` | DCGAN | | yes | | |
| `mbstd` | DCGAN | | | yes | |
| `dcgan-all` | DCGAN | yes | yes | yes | |
| `progan-fixed` | ProGAN | yes | yes | yes | |
| `progan-grow` | ProGAN | yes | yes | yes | yes |

Two comparisons carry the weight:

- **`dcgan-all` vs `progan-fixed`** separates "the three tricks bolted onto a
  DCGAN" from "ProGAN's actual architecture", with growing switched off in both.
- **`progan-fixed` vs `progan-grow`** is progressive growing on its own. The two
  arms are the *same network* — `test_the_two_progan_arms_are_architecturally_identical`
  asserts their parameter shapes match exactly — differing only in whether the
  step budget is spent coarse-to-fine.

---

## What is held fixed, and why it matters

Identical across every arm: step budget, batch size, optimiser and its betas,
learning rate, loss function, the EMA decay used for evaluation, the FID sample
count, the latent noise used for scoring, and the held-out reference images.

Three of those are worth spelling out because they are deviations from the
papers, made deliberately:

**The loss is the same everywhere.** DCGAN used the non-saturating logistic loss;
ProGAN used WGAN-GP. Changing the objective at the same time as the architecture
makes the comparison unreadable — you can no longer say whether growing helped or
whether WGAN-GP did. Every arm here uses the non-saturating loss. WGAN-GP and R1
are implemented in `src/pgan/losses.py` and reachable with `--loss wgan-gp` /
`--r1-gamma`, so the choice can be checked rather than argued about.

**Every arm gets an EMA generator.** ProGAN evaluates a running average of G's
weights; DCGAN does not. Giving it to only the ProGAN arms would hand them a free
advantage that has nothing to do with the architecture. Both the EMA and the raw
generator are scored, and both columns appear in RESULTS.md, so the effect of the
averaging is visible rather than hidden.

**Batch size does not change with resolution.** A real ProGAN run uses large
batches at 4x4 and small ones at 1024x1024. Varying it here would mean the
growing arm sees a different number of images than the others under the same step
count. It is pinned at 64 throughout.

**The step budget is fixed, not the epoch count.** The growing arm spends its
early steps at low resolution, which is precisely the paper's claim: the same
compute buys more when spent coarse-to-fine. Fixing steps rather than wall-clock
time is what makes that claim testable.

---

## File-by-file layout

```
src/pgan/
  config.py            The seven arms, defined once. Everything else reads them.
  schedule.py          step -> (stage, alpha). The growing schedule lives here alone.
  device.py            Device, TF32, and the bf16-over-fp16 choice.
  losses.py            Non-saturating, WGAN-GP, R1.
  train.py             The loop. One arm, one seed.
  evaluate.py          FID for a checkpoint, against cached reference features.
  smoke.py             Every arm, a few steps, synthetic data. No download needed.
  data/
    celeba.py          Memmap dataset + the train/holdout split.
    prepare.py         Parquet -> one (N,64,64,3) uint8 array.
    synthetic.py       Procedural stand-in, for tests only.
  models/
    layers.py          The four contributions, as isolated modules.
    dcgan.py           Baseline, with the components as flags. Arms 1-5.
    progan.py          ProGAN, with growing as a flag. Arms 6-7.
  metrics/
    fid.py             InceptionV3 features + Frechet distance.
scripts/
  fetch_celeba.py      Downloads the shards. Resumable.
  check_gpu.py         What fits, and how fast, before committing to a sweep.
  run_ablation.py      The whole sweep -> artifacts/ablation.json.
  make_grid.py         One row of samples per arm, same latents.
tests/                 74 tests. Layers, models, schedule, FID, data, training.
```

---

## Install

```bash
pixi install
```

Or without pixi:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

Check the GPU and see what batch size fits:

```bash
pixi run check-gpu
```

Prove the whole pipeline works before downloading anything — this trains all
seven arms for a few steps on procedurally generated data:

```bash
pixi run smoke
```

---

## Running it, in order

**1. Get the data** (1.39 GB, resumable, from the HuggingFace mirror — the
canonical Google Drive link rate-limits and breaks torchvision's downloader):

```bash
pixi run python scripts/fetch_celeba.py --dry-run   # sizes first, downloads nothing
pixi run fetch
```

**2. Decode it once** into a single memory-mapped array. 202,599 images as
separate files would mean 202,599 file opens per epoch:

```bash
pixi run prepare
```

Produces `data/celeba64.npy` (2.49 GB) and a `.json` beside it recording the
count, the crop geometry and the source.

**3. Train one arm**, to see it work:

```bash
pixi run python -m pgan.train dcgan --steps 12000
```

**4. Run the sweep**:

```bash
pixi run python scripts/run_ablation.py --seeds 0 1 2 --steps 12000
```

Writes `artifacts/ablation.json` with every per-seed number, not just the means,
so the table in RESULTS.md can be recomputed rather than taken on trust.

**5. Score a single checkpoint**, if you want to check one number by hand:

```bash
pixi run python -m pgan.evaluate checkpoints/dcgan_seed0/final.pt --n 10000
```

**6. Draw the sample sheet** — one row per arm, all from the same latents:

```bash
pixi run grid
```

---

## Cost on a 12 GB GPU

Measured with `scripts/check_gpu.py` on an RTX 4080 Laptop (11,851 MiB), bf16
autocast, batch 64:

| architecture | batch | peak VRAM | throughput |
|---|---|---|---|
| DCGAN | 64 | 0.23 GB | 53.1 it/s |
| DCGAN | 256 | 0.56 GB | 16.6 it/s |
| ProGAN | 64 | 2.08 GB | 4.9 it/s |
| ProGAN | 256 | 6.57 GB | 1.2 it/s |

ProGAN is an order of magnitude slower per step at the same batch size, and that
is not a bug: its blocks are two 3x3 convolutions at full resolution against
DCGAN's single strided transposed convolution, and it carries roughly three times
the parameters. This is worth knowing before starting a sweep — the ProGAN arms
dominate the wall clock.

Nothing here comes close to filling 12 GB. The bottleneck at 64x64 is kernel
launch overhead, not memory, which is why throughput *falls* above batch 64.

---

## Troubleshooting

**`FileNotFoundError: data/celeba64.npy`** — run `fetch` then `prepare`. The
error message says this too.

**The download stops partway.** `fetch_celeba.py` sends a Range header and
resumes from whatever is on disk. Just run it again.

**`prepare` is using several GB of RAM.** Expected. It decodes in chunks of 8,192
and concatenates once at the end; the peak is the final 2.49 GB array. It does
not stream to disk, because a second pass over 200k JPEGs to size the output
first costs more than the memory does.

**FID is enormous (in the hundreds) on a run that looks fine.** Check `--n`. FID
is biased upward at small sample counts, steeply below a few thousand. All the
numbers in RESULTS.md use the same `n`, which is recorded in the JSON.

**FID does not match a published number.** It will not, and it is not supposed
to. Different feature extractor. See the note at the top of `metrics/fid.py`.

**The growing arm's samples look blurry at the end of training.** Check the last
`res` and `alpha` values in `metrics.jsonl`. If the run ended mid-fade, the
generator is still blending in a half-trained block. The schedule reserves budget
after the final stage for exactly this reason; shortening `--steps` without
shortening `--steps-per-stage` breaks that.

---

## Honest limitations

**64x64 is the smallest resolution at which progressive growing means anything,
and possibly below it.** The paper's case is about the range between coarse and
fine structure. Five stages of 4 -> 64 is a short ladder. A negative result for
growing here is evidence about 64x64 and nothing larger.

**One dataset.** CelebA is aligned, centred, and unusually easy. Conclusions
about which components matter may not carry to unaligned or multi-class data.

**The step budget is short.** 12,000 steps at batch 64 is about 4 epochs of
CelebA. Some of these components may be about asymptotic behaviour rather than
early behaviour, and this budget cannot see that.

**FID is one metric and a blunt one.** It is sensitive to the feature extractor,
biased by sample count, and can be gamed by matching low-order statistics without
producing good images. The sample sheet from `make_grid.py` is there so the
numbers can be sanity-checked by eye.

**No hyperparameter search.** Learning rate, betas and EMA decay are the standard
values, held fixed. It is possible that an arm scoring badly here would do well
under settings tuned for it, and that possibility is not excluded.

---

## References

- Karras, Aila, Laine, Lehtinen. *Progressive Growing of GANs for Improved
  Quality, Stability, and Variation.* ICLR 2018.
- Radford, Metz, Chintala. *Unsupervised Representation Learning with Deep
  Convolutional Generative Adversarial Networks.* ICLR 2016.
- Heusel et al. *GANs Trained by a Two Time-Scale Update Rule Converge to a Local
  Nash Equilibrium.* NeurIPS 2017. (FID)
- Mescheder, Geiger, Nowozin. *Which Training Methods for GANs do actually
  Converge?* ICML 2018. (R1)
- Liu, Luo, Wang, Tang. *Deep Learning Face Attributes in the Wild.* ICCV 2015.
  (CelebA)
