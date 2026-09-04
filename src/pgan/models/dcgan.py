"""DCGAN baseline, with the ProGAN components switchable one at a time.

Radford et al. (2016) is the starting point: transposed convolutions in G,
strided convolutions in D, batch norm in both, LeakyReLU in D. Everything here
stays at a fixed 64x64, so the only thing changing between ablation arms is the
component under test and never the resolution schedule.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pgan.models.layers import (
    EqualizedConv2d,
    EqualizedConvTranspose2d,
    MinibatchStdDev,
    PixelNorm,
)

# 64x64 channel schedule. Halving the width each time the spatial size doubles
# keeps the parameter count roughly flat across blocks.
CHANNELS = [512, 256, 128, 64]


def _conv(eq, *args, **kw):
    return EqualizedConv2d(*args, **kw) if eq else nn.Conv2d(*args, **kw)


def _deconv(eq, *args, **kw):
    return EqualizedConvTranspose2d(*args, **kw) if eq else nn.ConvTranspose2d(*args, **kw)


class Generator(nn.Module):
    def __init__(self, z_dim=128, img_ch=3, equalized_lr=False, pixel_norm=False):
        super().__init__()
        self.z_dim = z_dim

        def norm(ch):
            # PixelNorm replaces batch norm rather than joining it. Stacking both
            # normalises twice and the batch statistics stop carrying information.
            return PixelNorm() if pixel_norm else nn.BatchNorm2d(ch)

        # 1x1 -> 4x4: kernel 4, stride 1, no padding is the standard DCGAN way of
        # turning the latent vector into a spatial feature map.
        blocks = [nn.Sequential(
            _deconv(equalized_lr, z_dim, CHANNELS[0], 4, 1, 0, bias=False),
            norm(CHANNELS[0]),
            nn.ReLU(inplace=True),
        )]
        for prev, cur in zip(CHANNELS, CHANNELS[1:]):
            blocks.append(nn.Sequential(
                _deconv(equalized_lr, prev, cur, 4, 2, 1, bias=False),
                norm(cur),
                nn.ReLU(inplace=True),
            ))
        self.blocks = nn.Sequential(*blocks)
        # gain 1.0 on the output layer: the tanh that follows saturates if the
        # pre-activations start out at the sqrt(2) scale used for ReLU blocks.
        self.to_rgb = _deconv(equalized_lr, CHANNELS[-1], img_ch, 4, 2, 1, gain=1.0) \
            if equalized_lr else nn.ConvTranspose2d(CHANNELS[-1], img_ch, 4, 2, 1)

        if not equalized_lr:
            self.apply(dcgan_init)

    def forward(self, z):
        if z.dim() == 2:
            z = z[:, :, None, None]
        return torch.tanh(self.to_rgb(self.blocks(z)))


class Discriminator(nn.Module):
    def __init__(self, img_ch=3, equalized_lr=False, minibatch_std=False):
        super().__init__()
        rev = CHANNELS[::-1]

        layers = [
            _conv(equalized_lr, img_ch, rev[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for prev, cur in zip(rev, rev[1:]):
            layers += [
                _conv(equalized_lr, prev, cur, 4, 2, 1, bias=False),
                # Batch norm is dropped when equalised learning rates are on: both
                # exist to hold activation scale steady, and the paper's
                # discriminator carries no normalisation at all.
                nn.Identity() if equalized_lr else nn.BatchNorm2d(cur),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.body = nn.Sequential(*layers)

        self.mbstd = MinibatchStdDev() if minibatch_std else None
        head_in = rev[-1] + (1 if minibatch_std else 0)
        self.head = _conv(equalized_lr, head_in, 1, 4, 1, 0, gain=1.0) if equalized_lr \
            else nn.Conv2d(head_in, 1, 4, 1, 0)

        if not equalized_lr:
            self.apply(dcgan_init)

    def forward(self, x):
        h = self.body(x)
        if self.mbstd is not None:
            h = self.mbstd(h)
        # Returns a raw logit per sample; the loss functions apply their own
        # activation so the same discriminator serves BCE and WGAN-GP alike.
        return self.head(h).flatten(1).mean(1)


def dcgan_init(m):
    """The initialisation from the DCGAN paper: N(0, 0.02) convs, N(1, 0.02) BN."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def build(cfg):
    g = Generator(cfg.z_dim, equalized_lr=cfg.equalized_lr, pixel_norm=cfg.pixel_norm)
    d = Discriminator(equalized_lr=cfg.equalized_lr, minibatch_std=cfg.minibatch_std)
    return g, d
