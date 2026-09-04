"""Train one arm of the ablation.

Everything that is not the component under test is held identical between arms:
the same step budget, batch size, optimiser, learning rate, loss, seed handling,
EMA decay and evaluation protocol. If any of those move, the arms stop being
comparable and the table in RESULTS.md means nothing.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pgan import losses
from pgan.config import ARMS, make
from pgan.data.celeba import CelebA64, infinite, to_uint8
from pgan.device import amp_dtype, autocast, describe, resolve, setup
from pgan.schedule import schedule, stages_for

EMA_DECAY = 0.999


def build_models(cfg):
    if cfg.arch == "dcgan":
        from pgan.models import dcgan
        return dcgan.build(cfg)
    from pgan.models import progan
    return progan.build(cfg)


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach(), 1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def real_at_stage(x, stage, alpha, stages):
    """Bring a full-resolution batch down to the stage the networks are at.

    During a fade the reals are blended the same way the generator's output is:
    part sharp, part the blurrier previous resolution upsampled back. Without
    this the discriminator sees crisp reals against half-upsampled fakes and can
    win on that alone, which stalls G for the whole fade window.
    """
    target = stages[stage]
    if x.shape[-1] != target:
        x = F.adaptive_avg_pool2d(x, target)
    if alpha < 1.0 and stage > 0:
        low = F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode="nearest")
        x = alpha * x + (1.0 - alpha) * low
    return x


def train(cfg, device=None, log=print):
    device = resolve(device)
    setup(device)
    torch.manual_seed(cfg.seed)

    dtype = amp_dtype(device, cfg.amp)
    g, d = build_models(cfg)
    g, d = g.to(device), d.to(device)
    # The EMA copy is what gets evaluated. Applied to every arm, not just the
    # ProGAN ones, so it cannot be the thing that separates them.
    g_ema = copy.deepcopy(g).eval()
    for p in g_ema.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.Adam(g.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    if cfg.data == "synthetic":
        # Test-only path. Nothing measured on this is reported anywhere.
        from pgan.data.synthetic import Blobs
        ds = Blobs(n=4096, size=cfg.resolution, seed=cfg.seed)
    else:
        ds = CelebA64(cfg.data, split="train")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=cfg.num_workers > 0)
    stream = infinite(loader)
    stages = stages_for(cfg.resolution)

    out = Path(cfg.out_dir) / f"{cfg.arm}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    metrics = (out / "metrics.jsonl").open("w")

    log(f"[runtime] {describe(device)}")
    log(f"[runtime] amp={dtype} arm={cfg.arm} arch={cfg.arch} grow={cfg.grow}")
    log(f"[data]    {len(ds):,} training images (holdout reserved for FID)")
    log(f"[model]   G {sum(p.numel() for p in g.parameters())/1e6:.2f}M  "
        f"D {sum(p.numel() for p in d.parameters())/1e6:.2f}M")

    fixed_z = torch.randn(64, cfg.z_dim, device=device)
    t0 = time.time()
    running = {"d": 0.0, "g": 0.0, "n": 0}

    for step in range(cfg.steps):
        stage, alpha = schedule(step, cfg)
        kw = {"stage": stage, "alpha": alpha} if cfg.arch == "progan" else {}

        real = next(stream).to(device, non_blocking=True)
        real = real_at_stage(real, stage, alpha, stages)

        # --- discriminator ---
        z = torch.randn(real.size(0), cfg.z_dim, device=device)
        with autocast(device, dtype):
            fake = g(z, **kw)
            d_real = d(real, **kw)
            d_fake = d(fake.detach(), **kw)
            if cfg.loss == "wgan-gp":
                loss_d = losses.d_loss_wgan(d_real, d_fake)
            else:
                loss_d = losses.d_loss_ns(d_real, d_fake)
        if cfg.loss == "wgan-gp":
            # The penalty is computed outside autocast: the double backward it
            # needs is numerically fragile in reduced precision.
            loss_d = loss_d + 10.0 * losses.gradient_penalty(
                d, real.float(), fake.detach().float(), kw)
        elif cfg.r1_gamma > 0:
            loss_d = loss_d + 0.5 * cfg.r1_gamma * losses.r1_penalty(d, real.float(), kw)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # --- generator ---
        z = torch.randn(real.size(0), cfg.z_dim, device=device)
        with autocast(device, dtype):
            fake = g(z, **kw)
            g_fake = d(fake, **kw)
            loss_g = losses.g_loss_wgan(g_fake) if cfg.loss == "wgan-gp" \
                else losses.g_loss_ns(g_fake)
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        ema_update(g_ema, g, EMA_DECAY)

        running["d"] += float(loss_d)
        running["g"] += float(loss_g)
        running["n"] += 1

        if (step + 1) % cfg.log_every == 0:
            n = running["n"]
            rec = {"step": step + 1, "stage": stage, "res": stages[stage],
                   "alpha": round(alpha, 4),
                   "loss_d": running["d"] / n, "loss_g": running["g"] / n,
                   "sec": round(time.time() - t0, 1)}
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            running = {"d": 0.0, "g": 0.0, "n": 0}
            if (step + 1) % (cfg.log_every * 10) == 0:
                log(f"  step {step+1:6d}/{cfg.steps}  res {stages[stage]:2d}  "
                    f"a {alpha:.2f}  d {rec['loss_d']:+.3f}  g {rec['loss_g']:+.3f}  "
                    f"{(step+1)/(time.time()-t0):.1f} it/s")

        if cfg.sample_every and (step + 1) % cfg.sample_every == 0:
            save_grid(g_ema, fixed_z, kw, out / f"samples_{step+1:06d}.png")

    metrics.close()
    elapsed = time.time() - t0
    ckpt = {"g": g.state_dict(), "g_ema": g_ema.state_dict(), "d": d.state_dict(),
            "cfg": cfg.to_dict(), "steps": cfg.steps, "seconds": elapsed}
    torch.save(ckpt, out / "final.pt")
    save_grid(g_ema, fixed_z, {}, out / "samples_final.png")
    msg = f"[done]    {cfg.steps} steps in {elapsed:.0f}s ({cfg.steps/elapsed:.1f} it/s)"
    if device.type == "cuda":
        msg += f", peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    log(msg)
    return out


@torch.no_grad()
def save_grid(g, z, kw, path, nrow=8):
    from torchvision.utils import save_image

    was = g.training
    g.eval()
    x = g(z, **kw).float()
    save_image(to_uint8(x).float().div(255), path, nrow=nrow)
    g.train(was)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=sorted(ARMS))
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--data")
    p.add_argument("--out-dir", dest="out_dir")
    p.add_argument("--loss", choices=["ns", "wgan-gp"])
    p.add_argument("--r1-gamma", dest="r1_gamma", type=float)
    p.add_argument("--steps-per-stage", dest="steps_per_stage", type=int)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    a = vars(p.parse_args())
    arm = a.pop("arm")
    device = a.pop("device")
    cfg = make(arm, **a)
    train(cfg, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
