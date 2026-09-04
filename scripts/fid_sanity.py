#!/usr/bin/env python
"""Calibrate the FID scale before trusting any number from it.

Two controls, both computed on real CelebA images only:

  floor    two disjoint halves of the held-out set against each other. This is
           the best score any generator could possibly achieve at this sample
           count -- it is not zero, because FID is biased upward on finite
           samples, and knowing how far from zero is what makes a generated
           score readable.

  ceiling  held-out images against the same images heavily corrupted. Gives a
           rough sense of what a bad-but-not-random score looks like.

Without the floor, a reported FID of 20 is a number with no scale attached.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from pgan.data.celeba import CelebA64, to_uint8
from pgan.device import describe, resolve, setup
from pgan.metrics.fid import InceptionFeatures, fid_from_features


def features(model, ds, idx, device, batch=100):
    out = []
    for i in range(0, len(idx), batch):
        chunk = torch.stack([ds[j] for j in idx[i:i + batch]])
        out.append(model(to_uint8(chunk).to(device)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/celeba64.npy")
    p.add_argument("--n", type=int, default=5000, help="images per side")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    device = resolve(a.device)
    setup(device)
    print(f"device: {describe(device)}")

    ds = CelebA64(a.data, split="holdout")
    if 2 * a.n > len(ds):
        raise SystemExit(f"need {2*a.n} holdout images, have {len(ds)}")
    model = InceptionFeatures().to(device).eval()

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ds))
    fa = features(model, ds, perm[:a.n], device)
    fb = features(model, ds, perm[a.n:2 * a.n], device)
    floor = fid_from_features(fa, fb)
    print(f"\n  floor   real vs real, disjoint, n={a.n:,}     FID {floor:8.3f}")

    # Heavy corruption: 8x downsample-upsample plus noise. Not a model, just a
    # marker for what "clearly wrong" scores.
    import torch.nn.functional as F
    imgs = torch.stack([ds[j] for j in perm[a.n:2 * a.n]])
    bad = F.interpolate(F.avg_pool2d(imgs, 8), scale_factor=8, mode="nearest")
    bad = (bad + 0.3 * torch.randn_like(bad)).clamp(-1, 1)
    fc = []
    for i in range(0, len(bad), 100):
        fc.append(model(to_uint8(bad[i:i + 100]).to(device)).float().cpu().numpy())
    ceiling = fid_from_features(fa, np.concatenate(fc))
    print(f"  ceiling real vs 8x-blurred + noise           FID {ceiling:8.3f}")
    print(f"\n  a generated score should land between these. Anything below the")
    print(f"  floor means the evaluation is comparing something to itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
