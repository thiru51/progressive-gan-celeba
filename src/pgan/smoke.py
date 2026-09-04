"""Run every arm for a handful of steps on synthetic data.

This is a plumbing check, not an experiment: it proves each arm builds, trains,
grows, saves and reloads without touching the 1.4 GB download. It should finish
in about a minute on a GPU. Nothing it prints is a result.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from pgan.config import ARMS, make
from pgan.device import describe, resolve
from pgan.train import build_models, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    device = resolve(a.device)
    print(f"device: {describe(device)}\n")

    with tempfile.TemporaryDirectory() as tmp:
        for arm in ARMS:
            cfg = make(arm, steps=a.steps, batch_size=a.batch_size,
                       data="synthetic", out_dir=tmp, num_workers=0,
                       # Four stages of growing inside 40 steps, so the fade
                       # logic is actually exercised rather than skipped.
                       steps_per_stage=max(2, a.steps // 6),
                       sample_every=0, log_every=a.steps)
            out = train(cfg, a.device, log=lambda *_: None)

            ck = torch.load(Path(out) / "final.pt", map_location="cpu", weights_only=False)
            g, _ = build_models(cfg)
            g.load_state_dict(ck["g"])
            with torch.no_grad():
                x = g(torch.randn(4, cfg.z_dim))
            assert x.shape == (4, 3, cfg.resolution, cfg.resolution), x.shape
            assert torch.isfinite(x).all(), f"{arm}: non-finite output"
            lines = (Path(out) / "metrics.jsonl").read_text().strip().splitlines()
            print(f"  {arm:14s} ok   out {tuple(x.shape)}  "
                  f"range[{x.min():+.2f},{x.max():+.2f}]  {len(lines)} log rows  "
                  f"{ck['seconds']:.1f}s")

    print("\nall arms build, train, checkpoint and reload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
