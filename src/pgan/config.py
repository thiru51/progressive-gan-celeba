"""One config object for every arm of the ablation.

The arms are defined here rather than in YAML so that the seven-row table in
RESULTS.md and the code that produced it cannot drift apart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # --- what is being tested ---
    arm: str = "dcgan"
    arch: str = "dcgan"            # "dcgan" or "progan"
    equalized_lr: bool = False
    pixel_norm: bool = False
    minibatch_std: bool = False
    grow: bool = False             # progressive growing; requires arch="progan"

    # --- held fixed across every arm ---
    resolution: int = 64
    z_dim: int = 128
    batch_size: int = 64
    steps: int = 12000
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    loss: str = "ns"               # "ns" (non-saturating logistic) or "wgan-gp"
    r1_gamma: float = 0.0          # R1 penalty on real samples; 0 disables it
    seed: int = 0

    # --- growing schedule, ignored unless grow=True ---
    # Steps spent at each resolution, half fading the new block in and half
    # stabilising it. The final resolution gets whatever budget is left, so the
    # total step count matches the non-growing arms exactly.
    steps_per_stage: int = 2000

    # --- data / bookkeeping ---
    data: str = "data/celeba64.npy"
    out_dir: str = "checkpoints"
    fid_every: int = 0             # 0 = only at the end
    sample_every: int = 2000
    log_every: int = 100
    num_workers: int = 4
    amp: bool = True

    def to_dict(self):
        return asdict(self)


# The seven arms. Read top to bottom this walks from DCGAN to ProGAN one change
# at a time, which is the only way to attribute the difference to anything.
ARMS = {
    "dcgan":         dict(arch="dcgan",  equalized_lr=False, pixel_norm=False, minibatch_std=False),
    "eqlr":          dict(arch="dcgan",  equalized_lr=True,  pixel_norm=False, minibatch_std=False),
    "pixelnorm":     dict(arch="dcgan",  equalized_lr=False, pixel_norm=True,  minibatch_std=False),
    "mbstd":         dict(arch="dcgan",  equalized_lr=False, pixel_norm=False, minibatch_std=True),
    "dcgan-all":     dict(arch="dcgan",  equalized_lr=True,  pixel_norm=True,  minibatch_std=True),
    "progan-fixed":  dict(arch="progan", equalized_lr=True,  pixel_norm=True,  minibatch_std=True, grow=False),
    "progan-grow":   dict(arch="progan", equalized_lr=True,  pixel_norm=True,  minibatch_std=True, grow=True),
}


def make(arm: str, **overrides) -> Config:
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    cfg = Config(arm=arm, **ARMS[arm])
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config field {k!r}")
        setattr(cfg, k, v)
    if cfg.grow and cfg.arch != "progan":
        raise ValueError("grow=True only makes sense with arch='progan'")
    return cfg
