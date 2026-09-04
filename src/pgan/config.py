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
    # Match the baseline's initial output-layer weight scale when equalised
    # learning rates are on. See the `eqlr-matched` arm and RESULTS.md: without
    # it the equalised generator starts with pre-tanh activations ~6.6x larger
    # than the baseline's, saturates its output head, and never recovers.
    match_out_scale: bool = False

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


# Learning rates.
#
# Under Adam the update magnitude is roughly the learning rate itself, because
# the optimiser divides out the gradient scale. What actually governs how fast a
# layer learns is therefore the *relative* step, lr / |w|. The two
# parameterisations here store weights at very different scales -- DCGAN
# initialises at N(0, 0.02), the equalised layers at N(0, 1) and apply He's
# constant at forward time -- so the same nominal learning rate produces
# relative steps that differ by a factor of 1/0.02 = 50.
#
# Holding the nominal learning rate fixed across arms would therefore not be a
# fair comparison; it would be a comparison at two different effective step
# sizes. These are set to hold the *effective* step fixed instead:
#
#     lr = BASE_LR / init_std_of_the_parameterisation
#
# The factor is derived, not tuned, and it was checked: sweeping the equalised
# arm over 2e-4 / 1e-3 / 3e-3 / 1e-2 gives FID 238 / 140 / 48 / 29 against the
# baseline's 26, recovering exactly at the predicted 50x. RESULTS.md has the
# table.
BASE_LR = 2e-4
DCGAN_INIT_STD = 0.02
EQUALIZED_LR = BASE_LR / DCGAN_INIT_STD   # 1e-2


# The eight arms. Read top to bottom this walks from DCGAN to ProGAN one change
# at a time, which is the only way to attribute the difference to anything.
ARMS = {
    "dcgan":         dict(arch="dcgan",  equalized_lr=False, pixel_norm=False, minibatch_std=False),
    # Deliberately left at the baseline learning rate: this arm is the naive
    # drop-in, and its failure is the point. Everything else equalised is run at
    # the matched effective step size.
    "eqlr":          dict(arch="dcgan",  equalized_lr=True,  pixel_norm=False, minibatch_std=False),
    "eqlr-matched":  dict(arch="dcgan",  equalized_lr=True,  pixel_norm=False, minibatch_std=False,
                          match_out_scale=True, lr_g=EQUALIZED_LR, lr_d=EQUALIZED_LR),
    "pixelnorm":     dict(arch="dcgan",  equalized_lr=False, pixel_norm=True,  minibatch_std=False),
    "mbstd":         dict(arch="dcgan",  equalized_lr=False, pixel_norm=False, minibatch_std=True),
    "dcgan-all":     dict(arch="dcgan",  equalized_lr=True,  pixel_norm=True,  minibatch_std=True,
                          match_out_scale=True, lr_g=EQUALIZED_LR, lr_d=EQUALIZED_LR),
    "progan-fixed":  dict(arch="progan", equalized_lr=True,  pixel_norm=True,  minibatch_std=True,
                          grow=False, lr_g=EQUALIZED_LR, lr_d=EQUALIZED_LR),
    "progan-grow":   dict(arch="progan", equalized_lr=True,  pixel_norm=True,  minibatch_std=True,
                          grow=True, lr_g=EQUALIZED_LR, lr_d=EQUALIZED_LR),
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
