#!/usr/bin/env python
"""Sweep one arm over learning rates and write the result as JSON.

This exists because the learning rate turned out not to be a free variable.
Under Adam the update magnitude is about the learning rate itself, so what
governs learning speed is the relative step, lr / |w|. The DCGAN and equalised
parameterisations store weights 50x apart, so the same nominal rate is not the
same experiment.

The factor is derivable. This script is what checked it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pgan.config import ARMS, make
from pgan.device import describe, resolve
from pgan.evaluate import score_checkpoint
from pgan.train import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=sorted(ARMS))
    p.add_argument("--lrs", nargs="+", type=float, default=[2e-4, 1e-3, 3e-3, 1e-2])
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fid-n", dest="fid_n", type=int, default=10000)
    p.add_argument("--data", default="data/celeba64.npy")
    p.add_argument("--r1-gamma", dest="r1_gamma", type=float, default=0.0)
    p.add_argument("--out-dir", dest="out_dir", default="/tmp/lr_probe")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    print(f"device: {describe(resolve(a.device))}")
    rows = []
    for lr in a.lrs:
        cfg = make(a.arm, seed=a.seed, steps=a.steps, data=a.data,
                   lr_g=lr, lr_d=lr, r1_gamma=a.r1_gamma,
                   out_dir=f"{a.out_dir}/{lr:g}")
        print(f"\n[{a.arm} lr={lr:g}] training")
        out = train(cfg, a.device)
        fid, _ = score_checkpoint(Path(out) / "final.pt", a.fid_n, a.data,
                                  a.device, "g_ema", log=lambda *_: None)
        rows.append({"lr": lr, "fid_ema": fid})
        print(f"[{a.arm} lr={lr:g}] FID {fid:.2f}")

    doc = {"arm": a.arm, "seed": a.seed, "steps": a.steps,
           "fid_samples": a.fid_n, "r1_gamma": a.r1_gamma,
           "device": describe(resolve(a.device)), "rows": rows}
    out_path = a.out or f"artifacts/lr_probe_{a.arm}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, indent=2))

    print(f"\n| lr | " + " | ".join(f"{r['lr']:g}" for r in rows) + " |")
    print("|---" * (len(rows) + 1) + "|")
    print("| FID | " + " | ".join(f"{r['fid_ema']:.1f}" for r in rows) + " |")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
