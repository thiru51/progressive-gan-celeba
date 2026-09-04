"""Score a trained checkpoint with FID against the held-out CelebA images.

The reference features are computed once and cached. Every arm is then scored
against the identical real statistics, which is the only way the arms can be
compared to each other at all.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pgan.config import make
from pgan.data.celeba import CelebA64, to_uint8
from pgan.device import describe, resolve, setup
from pgan.metrics.fid import InceptionFeatures, fid_from_features
from pgan.train import build_models

CACHE = Path("artifacts/real_features.npz")


def real_features(model, device, data, n, batch_size=64, cache=CACHE, log=print):
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache)
        if int(z["n"]) >= n and str(z["data"]) == str(data):
            log(f"  reference features from cache ({int(z['n']):,} images)")
            return z["feats"][:n]

    ds = CelebA64(data, split="holdout")
    if n > len(ds):
        raise ValueError(f"asked for {n} reference images, holdout has {len(ds)}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    feats, seen, t0 = [], 0, time.time()
    for batch in loader:
        feats.append(model(to_uint8(batch).to(device)).float().cpu().numpy())
        seen += batch.shape[0]
        if seen >= n:
            break
    out = np.concatenate(feats)[:n]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, feats=out, n=out.shape[0], data=str(data))
    log(f"  reference features: {out.shape[0]:,} images in {time.time()-t0:.0f}s -> {cache}")
    return out


@torch.no_grad()
def fake_features(model, generator, device, n, z_dim, batch_size=64, seed=1234):
    """Sample n images and featurise them.

    The seed is fixed and separate from the training seed so that two scorings
    of the same checkpoint agree exactly, and so that the sample noise is common
    across arms rather than an extra source of difference between them.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    feats, made = [], 0
    while made < n:
        b = min(batch_size, n - made)
        z = torch.randn(b, z_dim, generator=gen).to(device)
        x = generator(z).float()
        feats.append(model(to_uint8(x)).float().cpu().numpy())
        made += b
    return np.concatenate(feats)[:n]


def score_checkpoint(ckpt_path, n=10000, data=None, device=None, which="g_ema", log=print):
    device = resolve(device)
    setup(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = make(ck["cfg"]["arm"], **{k: v for k, v in ck["cfg"].items()
                                    if k not in ("arm",)})
    g, _ = build_models(cfg)
    g.load_state_dict(ck[which])
    g = g.to(device).eval()

    inception = InceptionFeatures().to(device).eval()
    real = real_features(inception, device, data or cfg.data, n, log=log)
    fake = fake_features(inception, g, device, n, cfg.z_dim)
    return fid_from_features(real, fake), cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--n", type=int, default=10000,
                   help="samples per side; FID is biased upward at small n")
    p.add_argument("--data", default=None)
    p.add_argument("--which", default="g_ema", choices=["g", "g_ema"])
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    print(f"device: {describe(resolve(a.device))}")
    t0 = time.time()
    fid, cfg = score_checkpoint(a.checkpoint, a.n, a.data, a.device, a.which)
    print(f"  arm {cfg.arm}  seed {cfg.seed}  weights {a.which}  n {a.n:,}")
    print(f"  FID {fid:.3f}   ({time.time()-t0:.0f}s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"checkpoint": str(a.checkpoint), "arm": cfg.arm, "seed": cfg.seed,
             "weights": a.which, "n": a.n, "fid": fid}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
