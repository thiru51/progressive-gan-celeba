#!/usr/bin/env python
"""Render the RESULTS.md tables straight from artifacts/ablation.json.

The point is that the numbers in the write-up are generated from the artifact
rather than retyped from it. Retyped numbers drift; generated ones cannot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Presentation order: the walk from DCGAN to ProGAN, not best-first.
ORDER = ["dcgan", "eqlr", "eqlr-matched", "pixelnorm", "mbstd",
         "dcgan-all", "progan-fixed", "progan-grow"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default="artifacts/ablation.json")
    p.add_argument("--floor", type=float, default=None,
                   help="real-vs-real FID from scripts/fid_sanity.py")
    a = p.parse_args()

    d = json.loads(Path(a.json).read_text())
    arms = d["arms"]
    base = arms.get("dcgan", {}).get("fid_ema_mean")

    print(f"Run on {d['device']}")
    print(f"{d['steps']:,} steps, batch {d['batch_size']}, seeds {d['seeds']}, "
          f"FID at n={d['fid_samples']:,}\n")

    print("| arm | FID (EMA) | FID (raw) | vs baseline | train s |")
    print("|---|---|---|---|---|")
    for arm in ORDER:
        if arm not in arms:
            continue
        r = arms[arm]
        delta = ""
        if base is not None and r["fid_ema_mean"] is not None:
            diff = r["fid_ema_mean"] - base
            delta = "baseline" if arm == "dcgan" else f"{diff:+.2f}"
        secs = sum(r["seconds"]) / len(r["seconds"]) if r["seconds"] else 0
        print(f"| `{arm}` | {r['fid_ema_mean']:.2f} +-{r['fid_ema_std']:.2f} "
              f"| {r['fid_raw_mean']:.2f} +-{r['fid_raw_std']:.2f} | {delta} | {secs:.0f} |")

    print("\nPer-seed FID (EMA), so the means above can be checked:\n")
    print("| arm | " + " | ".join(f"seed {s}" for s in d["seeds"]) + " |")
    print("|---" * (len(d["seeds"]) + 1) + "|")
    for arm in ORDER:
        if arm not in arms:
            continue
        vals = " | ".join(f"{v:.2f}" for v in arms[arm]["fid_ema"])
        print(f"| `{arm}` | {vals} |")

    if a.floor is not None:
        print(f"\nReal-vs-real floor at this sample count: {a.floor:.2f}. "
              f"No generator can score below it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
