#!/usr/bin/env python
"""Build the side-by-side sample sheet used in RESULTS.md.

One row of samples per arm, all drawn from the same latent codes, so the rows
differ because the models differ and not because the noise did.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pgan.config import ARMS, make
from pgan.data.celeba import to_uint8
from pgan.device import resolve, setup
from pgan.train import build_models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=8, help="samples per arm")
    p.add_argument("--latent-seed", dest="latent_seed", type=int, default=7)
    p.add_argument("--which", default="g_ema", choices=["g", "g_ema"])
    p.add_argument("--out", default="results/samples_by_arm.png")
    a = p.parse_args()

    device = resolve(None)
    setup(device)
    root = Path(a.checkpoints)

    gen = torch.Generator().manual_seed(a.latent_seed)
    rows, labels = [], []
    for arm in ARMS:
        ckpt = root / f"{arm}_seed{a.seed}" / "final.pt"
        if not ckpt.exists():
            print(f"  skip {arm}: no {ckpt}")
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = make(arm, **{k: v for k, v in ck["cfg"].items() if k != "arm"})
        g, _ = build_models(cfg)
        g.load_state_dict(ck[a.which])
        g = g.to(device).eval()
        z = torch.randn(a.n, cfg.z_dim, generator=gen).to(device)
        with torch.no_grad():
            rows.append(to_uint8(g(z).float()).cpu())
        labels.append(arm)
        print(f"  {arm}")

    if not rows:
        print("nothing to draw")
        return 1

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), a.n, figsize=(a.n * 1.1, len(rows) * 1.25))
    axes = axes.reshape(len(rows), a.n)
    for r, (batch, name) in enumerate(zip(rows, labels)):
        for c in range(a.n):
            ax = axes[r, c]
            ax.imshow(batch[c].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(f"same latents, seed {a.seed}, {a.which} weights", fontsize=9)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
