# Results

Run 4 September 2026. Every number here comes from `artifacts/ablation.json`,
written by `scripts/run_ablation.py`, and every mean was recomputed from the
per-seed values before being written down. Nothing is estimated.

## Read this first

**FID here is not FID anywhere else.** The features come from torchvision's
ImageNet InceptionV3, not the TF-Slim graph (`pt_inception-2015-12-05`) that
`pytorch-fid` and every published FID table use. Different weights, different
scale. These numbers are comparable to each other and to nothing else. Do not
put one on a slide next to the ProGAN paper's.

**This is an attribution study at 64x64, not a reproduction.** ProGAN trains to
1024x1024 for days. This is 12,000 steps at 64x64 on a laptop GPU. The
conclusions are about what these components do *under this budget at this
resolution*, and the growing result in particular should not be read as a claim
about the paper.

## What was run

| | |
|---|---|
| Data | 202,599 CelebA aligned faces at 64x64 |
| Reference | **20,000 held out**, never trained on; FID scored against 10,000 of them |
| Arms | 8, differing in one component at a time |
| Seeds | 0, 1, 2 per arm -- 24 runs |
| Budget | 12,000 generator steps, batch 64, identical for every arm |
| Loss | non-saturating logistic + R1 (gamma 10, lazy every 16), betas (0, 0.99) |
| Evaluation | EMA generator (decay 0.999); the raw generator is also scored |
| Hardware | RTX 4080 Laptop, 12 GB, bf16 autocast |
| Wall clock | 2.4 h for the sweep (plus the configuration search that preceded it) |

## Calibration: what the scale means

Three reference points, all measured, without which a FID of 30 is a number with
no meaning attached:

| | FID | what it is |
|---|---|---|
| **floor** | **2.29** | real vs real, two disjoint halves of the holdout, n=10,000. FID is biased upward on finite samples, so this -- not zero -- is the best any generator could score here. |
| **ceiling** | **284.06** | real vs the same images 8x downsampled and noised. A marker for "clearly broken". |
| **noise** | **~1.9** | two runs of `dcgan` at **the same seed and the same config** scored 29.89 and 31.80. cuDNN autotuning and TF32 make GPU training non-bit-reproducible. |

**That last row is the important one.** Any gap below about 2 FID is not
evidence of anything, even between two arms at the same seed. It is applied
throughout what follows.

## The headline

| arm | FID (EMA) | FID (raw) | vs baseline | train s |
|---|---|---|---|---|
| `dcgan` | 31.51 +-1.75 | 37.80 +-3.66 | baseline | 158 |
| `eqlr` | 293.30 +-4.35 | 294.58 +-1.98 | +261.79 | 162 |
| `eqlr-matched` | **23.73** +-1.97 | 29.09 +-4.18 | **-7.78** | 160 |
| `pixelnorm` | 31.15 +-0.81 | 41.43 +-1.65 | -0.36 | 168 |
| `mbstd` | 31.71 +-1.24 | 37.85 +-3.54 | +0.20 | 161 |
| `dcgan-all` | **22.70** +-0.22 | 31.76 +-0.71 | **-8.80** | 182 |
| `progan-fixed` | 28.38 +-0.82 | 31.26 +-3.82 | -3.13 | 2119 |
| `progan-grow` | 54.31 +-2.72 | 59.85 +-5.51 | **+22.80** | 1123 |

Per-seed, so the means can be checked rather than trusted:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `dcgan` | 29.89 | 33.95 | 30.69 |
| `eqlr` | 294.88 | 297.67 | 287.36 |
| `eqlr-matched` | 26.48 | 22.74 | 21.96 |
| `pixelnorm` | 32.00 | 31.38 | 30.06 |
| `mbstd` | 33.43 | 31.14 | 30.56 |
| `dcgan-all` | 22.82 | 22.90 | 22.40 |
| `progan-fixed` | 29.55 | 27.78 | 27.81 |
| `progan-grow` | 53.31 | 51.58 | 58.03 |

`results/samples_by_arm.png` shows eight samples per arm from identical latent
codes. The ordering by eye matches the ordering by FID, which is the check that
the metric is measuring something real here.

## What the numbers support

### 1. The equalised learning rate does essentially all of the work

`eqlr-matched` alone reaches 23.73 against the baseline's 31.51. `dcgan-all` --
that plus pixel norm plus minibatch stddev -- reaches 22.70. The gap between
them is **1.02 FID, inside the 1.9 noise floor**, so on this evidence the other
two components add nothing detectable once the equalised learning rate is in
place.

### 2. Pixel norm and minibatch stddev do nothing measurable on their own

| arm | FID | vs baseline |
|---|---|---|
| `pixelnorm` | 31.15 | -0.36 |
| `dcgan` | 31.51 | -- |
| `mbstd` | 31.71 | +0.20 |

All three inside 0.6 FID of each other, well under the noise floor. This is a
clean negative result: **neither component, added alone to a DCGAN, changes
sample quality at this budget.**

That does not mean they do nothing in general. Minibatch stddev exists to
prevent mode collapse, and no arm here collapsed -- the failure it defends
against never occurred, so there was nothing for it to prevent. Testing it
properly needs a setting that actually collapses.

### 3. Progressive growing makes things substantially worse

**This is the headline, and it is negative.**

| | FID |
|---|---|
| `progan-fixed` | 28.38 +-0.82 |
| `progan-grow` | 54.31 +-2.72 |

**+25.93 FID, 1.91x worse, on the identical network.** The two arms share the
same parameters, the same channel schedule, the same optimiser, the same step
budget and the same seeds. `test_the_two_progan_arms_are_architecturally_identical`
asserts their parameter shapes match. The *only* difference is whether the step
budget is spent coarse-to-fine. All three seeds agree (51.6, 53.3, 58.0) and the
gap is thirteen times the noise floor.

The likely reason is visible in the schedule: at 64x64 the growing arm spends
10,000 of its 12,000 steps below full resolution and reaches 64x64 only for the
last 2,000. The paper's argument -- settle coarse structure cheaply before
spending capacity on detail -- needs a long ladder to pay off, and 4 -> 64 is
five rungs. Under a **fixed step budget** at this resolution, growing is a cost
rather than a saving.

Two things this does **not** show. It does not show growing is worthless: the
paper's regime is 1024x1024, where the ladder is twice as long and the low
resolutions are genuinely much cheaper per step. And a fixed *step* budget is
not a fixed *compute* budget -- the growing arm finished in 1,123 s against
2,119 s, so it was 1.9x cheaper in wall clock. Given equal wall clock rather
than equal steps it would get roughly twice the updates, and this comparison
does not tell you where that lands.

### 4. ProGAN's architecture is not better than a DCGAN carrying the same three tricks

`progan-fixed` (28.38) against `dcgan-all` (22.70): the purpose-built
architecture is **5.7 FID worse** than a DCGAN with equalised learning rates,
pixel norm and minibatch stddev bolted on -- while costing 13x the training time
(2,119 s against 182 s). At 64x64 its extra depth and its two-3x3-convolutions
per block buy nothing.

### 5. Equalised learning rates are not a drop-in change

`eqlr` -- the same weight scaling dropped into a DCGAN with no other adjustment
-- scores **293.30**, above the corruption ceiling of 284. Its samples are
noise. This arm is kept deliberately as a control, because the failure is
instructive and it is the mistake anyone reimplementing the paper would make.

Two separate causes, both measured:

- **The output head starts saturated.** He's gain on the final layer gives
  pre-tanh activations of std 1.46 against the baseline's 0.22, so 8% of every
  generated image is pinned at +-1 from step zero and the gradient reaching the
  generator's first layer is 7x smaller. `match_out_scale` reproduces DCGAN's
  own initial weight scale instead: std 0.24, zero saturation.
- **The effective step size drops ~50x.** Under Adam the update magnitude is
  about the learning rate itself, so what governs learning is the relative step
  `lr / |w|`. DCGAN stores weights at N(0, 0.02) and the equalised layers at
  N(0, 1). Measured: 0.098 against 0.0018 relative movement over 20 steps.

Correcting both -- and only then -- gives `eqlr-matched` at 23.73. The learning
rate correction was *derived* (2e-4 / 0.02 = 1e-2) and then confirmed:

| lr | 2e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|
| FID | 238.1 | 139.9 | 48.3 | **29.1** |

**Where the derivation stops working.** The same argument predicts every
equalised network wants 1e-2. `progan-fixed` wants 1e-3 and scores 305.6 at
1e-2. Two fully equalised architectures, learning rates 10x apart, and the
difference is whether the discriminator is normalised. The weight-scale
prediction is right for a normalised critic and wrong for an unnormalised one.

### 6. ProGAN's architecture needs its regulariser

The first version of this experiment used the non-saturating loss with no
gradient penalty for every arm, on the grounds that changing the objective
alongside the architecture makes the result unattributable. That reasoning is
sound and the consequence was still fatal: ProGAN's discriminator carries no
normalisation at all -- which is exactly why the paper pairs it with WGAN-GP --
and removing the penalty left nothing constraining it. Every ProGAN run
collapsed, at every learning rate tried.

Adding the paper's pieces back, `progan-fixed` at 4,000 steps:

| | FID |
|---|---|
| lr 2e-4, betas (0.5, 0.999), no penalty | 364.0 |
| lr 1e-3 | 180.8 |
| + R1, gamma 10 | 147.6 |
| + betas (0, 0.99) | **75.2** |

R1 and betas `(0, 0.99)` are therefore shared defaults for **every** arm, so
they cannot be what separates them. The check that this was safe: the DCGAN
baseline barely moved, 25.6 -> 28.4.

## Honest limits

**The configuration was chosen with knowledge of the outcomes.** This is the
real weakness. R1, the betas and the per-architecture learning rate were not
fixed in advance; they were arrived at after arms collapsed, by diagnosing why
and correcting it, then applying the correction uniformly. The learning-rate
search is reported above rather than buried, but this is **not a pre-registered
experiment and should not be read as one**.

Concretely: **seed 0 carried every configuration decision**, so its numbers are
not independent of them. Seeds 1 and 2 ran afterwards with the configuration
frozen. The per-seed table is above so this can be inspected -- and seed 0 is
not systematically the best seed, which is mildly reassuring but not proof.

**Learning rate is tuned per architecture, not per arm.** Within the DCGAN body
all components are compared at a common rate, and within the ProGAN body
likewise, but the two bodies differ. An arm scoring badly might do better under
settings tuned for it specifically.

**64x64 is close to the smallest resolution at which growing could help.** The
negative result is evidence about 64x64 and nothing larger. 128x128 fits in
12 GB and is the run that would actually test the paper's claim.

**Fixed steps, not fixed compute.** Noted above for growing; it applies to
`progan-fixed` vs `dcgan-all` too, in the opposite direction -- ProGAN is 13x
slower per run, so under equal wall clock it would look worse still.

**One dataset, one metric, one budget.** CelebA is aligned and unusually easy.
FID is blunt, gameable by matching low-order statistics, and sensitive to the
feature extractor. 12,000 steps is about 4 epochs; components that matter
asymptotically would not show up.

**Three seeds is few.** The standard deviations above are population stddev over
n=3. They indicate spread, not a confidence interval.

## Verifying the results

`artifacts/ablation.json` holds every per-seed number, not just the means, so
the table can be recomputed:

```python
import json, statistics
d = json.load(open("artifacts/ablation.json"))
statistics.mean(d["arms"]["progan-grow"]["fid_ema"])   # 54.31
```

To re-derive from scratch (about 2.4 h on a 12 GB GPU):

```bash
python scripts/fetch_celeba.py && python -m pgan.data.prepare
python scripts/run_ablation.py --seeds 0 1 2 --steps 12000
```

To check the scale before trusting any of it:

```bash
python scripts/fid_sanity.py --n 10000     # floor 2.29, ceiling 284.06
```

**What is committed:** all code, 79 tests, the config and per-interval
`metrics.jsonl` for all 24 runs, the sample sheets, and `artifacts/ablation.json`.
**What is not:** the dataset (2.49 GB, two commands to rebuild), the weights
(~40 MB per DCGAN run, ~132 MB per ProGAN run, 24 runs), and the cached
Inception features (74 MB, regenerable). Reproducing the table means retraining;
reading the evidence does not.

## Summary

Under a fixed 12,000-step budget at 64x64 on CelebA, of ProGAN's four
contributions **only the equalised learning rate measurably improves on a DCGAN
baseline** (31.51 -> 23.73). Pixel norm and minibatch stddev change nothing
detectable on their own. **Progressive growing makes results substantially
worse** -- 54.31 against 28.38 on the identical network, consistent across three
seeds and thirteen times the noise floor -- because at this resolution it spends
most of the budget below full resolution and the coarse-to-fine ladder is too
short to repay that.

The equalised learning rate is also the component most easily got wrong: dropped
in naively it produces pure noise (FID 293), and making it work requires
matching the output-layer initialisation scale and rescaling the learning rate
by the 50x factor the parameterisation implies.
