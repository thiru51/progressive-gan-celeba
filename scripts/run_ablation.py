#!/usr/bin/env python
"""Run every arm at every seed and write one JSON with all of it.

Order matters here: arms are run seed-major (all arms at seed 0, then all at
seed 1) so that an interrupted sweep still has a complete, comparable set of
arms rather than three seeds of one arm and none of the rest.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from pgan.config import ARMS, make
from pgan.device import describe, resolve
from pgan.evaluate import score_checkpoint
from pgan.train import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--fid-n", dest="fid_n", type=int, default=10000)
    p.add_argument("--data", default="data/celeba64.npy")
    p.add_argument("--out-dir", dest="out_dir", default="checkpoints")
    p.add_argument("--out", default="artifacts/ablation.json")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-trained", action="store_true",
                   help="reuse an existing final.pt instead of retraining")
    a = p.parse_args()

    device = resolve(a.device)
    print(f"device: {describe(device)}")
    print(f"arms  : {', '.join(a.arms)}")
    print(f"seeds : {a.seeds}   steps: {a.steps}   batch: {a.batch_size}\n")

    results = {arm: {"fid_ema": [], "fid_raw": [], "seconds": []} for arm in a.arms}
    t_start = time.time()

    for seed in a.seeds:
        for arm in a.arms:
            cfg = make(arm, seed=seed, steps=a.steps, batch_size=a.batch_size,
                       data=a.data, out_dir=a.out_dir)
            out = Path(a.out_dir) / f"{arm}_seed{seed}"
            ckpt = out / "final.pt"
            if a.skip_trained and ckpt.exists():
                print(f"[{arm} seed{seed}] reusing {ckpt}")
            else:
                print(f"[{arm} seed{seed}] training")
                train(cfg, a.device)

            secs = torch.load(ckpt, map_location="cpu", weights_only=False)["seconds"]
            fid_ema, _ = score_checkpoint(ckpt, a.fid_n, a.data, a.device, "g_ema",
                                          log=lambda *_: None)
            fid_raw, _ = score_checkpoint(ckpt, a.fid_n, a.data, a.device, "g",
                                          log=lambda *_: None)
            results[arm]["fid_ema"].append(fid_ema)
            results[arm]["fid_raw"].append(fid_raw)
            results[arm]["seconds"].append(secs)
            print(f"[{arm} seed{seed}] FID ema {fid_ema:.2f}  raw {fid_raw:.2f}  "
                  f"({secs:.0f}s train)\n")

    for arm, r in results.items():
        for key in ("fid_ema", "fid_raw"):
            vals = r[key]
            r[key + "_mean"] = statistics.mean(vals) if vals else None
            r[key + "_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    doc = {
        "device": describe(device),
        "steps": a.steps, "batch_size": a.batch_size, "seeds": a.seeds,
        "fid_samples": a.fid_n,
        "fid_note": ("torchvision ImageNet InceptionV3 features, not the TF-Slim "
                     "graph used in published FID tables; comparable within this "
                     "repo only"),
        "reference": "held-out CelebA images, never seen in training",
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "arms": results,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2))

    print(f"\n{'arm':14s} {'FID (EMA)':>18s} {'FID (raw)':>18s}")
    order = sorted(results, key=lambda k: results[k]["fid_ema_mean"] or 1e9)
    for arm in order:
        r = results[arm]
        print(f"{arm:14s} {r['fid_ema_mean']:11.2f} +-{r['fid_ema_std']:5.2f} "
              f"{r['fid_raw_mean']:11.2f} +-{r['fid_raw_std']:5.2f}")
    print(f"\nwrote {a.out}  ({doc['wall_clock_seconds']:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
