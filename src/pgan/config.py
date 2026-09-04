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
    # (0, 0.99) rather than DCGAN's (0.5, 0.999). ProGAN's discriminator carries
    # no normalisation, and with momentum on the first moment it runs away: the
    # same run scores FID 148 at (0.5, 0.999) and 75 at (0, 0.99). Applied to
    # every arm so it cannot be what separates them.
    beta1: float = 0.0
    beta2: float = 0.99
    loss: str = "ns"               # "ns" (non-saturating logistic) or "wgan-gp"
    # R1 gradient penalty on real samples. Needed because an unnormalised
    # discriminator is otherwise unconstrained -- without it the ProGAN arms
    # collapse (FID 181 -> 148 at gamma 10). Applied to every arm.
    r1_gamma: float = 10.0
    # Lazy regularisation (StyleGAN2 s.4): the penalty every r1_every steps with
    # gamma scaled to match, rather than every step. R1 needs a double backward
    # and costs ~45% of throughput when applied every step; every 16 recovers
    # nearly all of it and the training dynamics are indistinguishable.
    r1_every: int = 16
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
# The derived factor is a first-order prediction and it does not survive
# contact with the ProGAN architecture. Selected on seed 0 at a fixed budget,
# everything else held constant:
#
#   eqlr-matched (DCGAN body)   1e-3 -> FID 190.7   1e-2 -> FID  23.0
#   progan-fixed (ProGAN body)  1e-3 -> FID  74.8   1e-2 -> FID 305.6
#
# The two equalised architectures want learning rates 10x apart. The difference
# is the discriminator: the DCGAN body keeps batch norm, the ProGAN body has no
# normalisation anywhere, and an unnormalised critic will not tolerate the
# larger step. 1e-3 is also the value the ProGAN paper uses, which is some
# comfort that this is the architecture's property and not an artefact of this
# particular budget.
PROGAN_LR = 1e-3


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
                          grow=False, lr_g=PROGAN_LR, lr_d=PROGAN_LR),
    "progan-grow":   dict(arch="progan", equalized_lr=True,  pixel_norm=True,  minibatch_std=True,
                          grow=True, lr_g=PROGAN_LR, lr_d=PROGAN_LR),
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
