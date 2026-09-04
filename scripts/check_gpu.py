#!/usr/bin/env python
"""What the machine is, and what batch size fits. Run this before a sweep."""
from __future__ import annotations

import torch

from pgan.config import make
from pgan.device import amp_dtype, autocast, describe, resolve, setup
from pgan.train import build_models


def main():
    device = resolve(None)
    setup(device)
    print(describe(device))
    if device.type != "cuda":
        print("no CUDA device; everything below is skipped")
        return 0

    dtype = amp_dtype(device, True)
    print(f"autocast dtype: {dtype}\n")
    print(f"{'arm':14s} {'batch':>6s} {'peak GB':>9s} {'it/s':>7s}")
    for arm in ("dcgan", "progan-fixed"):
        for bs in (32, 64, 128, 256):
            cfg = make(arm, batch_size=bs)
            g, d = build_models(cfg)
            g, d = g.to(device), d.to(device)
            og = torch.optim.Adam(g.parameters(), lr=1e-4)
            od = torch.optim.Adam(d.parameters(), lr=1e-4)
            torch.cuda.reset_peak_memory_stats()
            try:
                import time
                t0 = time.time()
                for _ in range(12):
                    z = torch.randn(bs, cfg.z_dim, device=device)
                    with autocast(device, dtype):
                        fake = g(z)
                        loss_d = d(fake.detach()).mean()
                    od.zero_grad(set_to_none=True); loss_d.backward(); od.step()
                    with autocast(device, dtype):
                        loss_g = -d(g(z)).mean()
                    og.zero_grad(set_to_none=True); loss_g.backward(); og.step()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"{arm:14s} {bs:6d} {peak:9.2f} {12/(time.time()-t0):7.1f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{arm:14s} {bs:6d} {'OOM':>9s} {'-':>7s}")
            finally:
                del g, d, og, od
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
