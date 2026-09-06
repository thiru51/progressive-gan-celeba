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
    p.add_argument("--resolution", type=int, default=64,
                   help="64 or 128. Must match the prepared dataset.")
    p.add_argument("--steps-per-stage", dest="steps_per_stage", type=int, default=2000,
                   help="growing budget per stage. At 128 there are six stages, so this "
                        "must be small enough that growing finishes inside --steps.")
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
                       data=a.data, out_dir=a.out_dir, resolution=a.resolution,
                       steps_per_stage=a.steps_per_stage)
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
                  f"({secs:.0f}s train)\n", flush=True)
            # Written after every run, not just at the end. A sweep is hours
            # long; losing power at hour four should cost the run in flight,
            # not the whole table. (It has already happened once.)
            save(a, results, device, t_start, partial=True)

    save(a, results, device, t_start, partial=False)
    print(f"\n{'arm':14s} {'FID (EMA)':>18s} {'FID (raw)':>18s}")
    order = sorted([k for k in results if results[k]["fid_ema"]],
                   key=lambda k: statistics.mean(results[k]["fid_ema"]))
    for arm in order:
        r = results[arm]
        print(f"{arm:14s} {statistics.mean(r['fid_ema']):11.2f} "
              f"+-{statistics.pstdev(r['fid_ema']) if len(r['fid_ema'])>1 else 0.0:5.2f} "
              f"{statistics.mean(r['fid_raw']):11.2f} "
              f"+-{statistics.pstdev(r['fid_raw']) if len(r['fid_raw'])>1 else 0.0:5.2f}")
    print(f"\nwrote {a.out}")
    return 0


def save(a, results, device, t_start, partial):
    out = {k: dict(v) for k, v in results.items()}
    for arm, r in out.items():
        for key in ("fid_ema", "fid_raw"):
            vals = r[key]
            r[key + "_mean"] = statistics.mean(vals) if vals else None
            r[key + "_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    doc = {
        "complete": not partial,
        "device": describe(device),
        "steps": a.steps, "batch_size": a.batch_size, "seeds": a.seeds,
        "resolution": a.resolution, "steps_per_stage": a.steps_per_stage,
        "fid_samples": a.fid_n,
        "fid_note": ("torchvision ImageNet InceptionV3 features, not the TF-Slim "
                     "graph used in published FID tables; comparable within this "
                     "repo only"),
        "reference": "held-out CelebA images, never seen in training",
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "arms": out,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename: a crash mid-write would otherwise leave
    # a truncated JSON that looks like a finished sweep.
    tmp = Path(str(a.out) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(Path(a.out))


if __name__ == "__main__":
    raise SystemExit(main())
