"""DCGAN baseline, with the ProGAN components switchable one at a time.

Radford et al. (2016) is the starting point: transposed convolutions in G,
strided convolutions in D, batch norm in both, LeakyReLU in D. Everything here
stays at a fixed 64x64, so the only thing changing between ablation arms is the
component under test and never the resolution schedule.
"""
from __future__ import annotations

import math

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


# The std DCGAN initialises every convolution with. The equalised output layer
# is matched to it in the `eqlr-matched` arm.
DCGAN_INIT_STD = 0.02


class Generator(nn.Module):
    def __init__(self, z_dim=128, img_ch=3, equalized_lr=False, pixel_norm=False,
                 match_out_scale=False):
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
        # The output layer's gain decides whether the tanh saturates at step 0.
        # He's gain of 1.0 gives pre-tanh std ~1.46 against the baseline's 0.22,
        # which saturates 8% of pixels and costs the generator ~7x of its
        # gradient. match_out_scale instead picks the gain that reproduces
        # DCGAN's own initial weight scale, so the two arms start from the same
        # place and the comparison is about learning, not about initialisation.
        out_gain = 1.0
        if match_out_scale:
            fan_in = img_ch * 4 * 4
            out_gain = DCGAN_INIT_STD * math.sqrt(fan_in)
        self.to_rgb = _deconv(equalized_lr, CHANNELS[-1], img_ch, 4, 2, 1, gain=out_gain) \
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
                # Batch norm stays in regardless of the equalised-LR flag. An
                # earlier version dropped it here on the grounds that ProGAN's
                # discriminator carries no normalisation -- which is true, but it
                # made the `eqlr` arm change two things at once, and the whole
                # point of this repository is that it changes one. Removing
                # normalisation belongs to the `progan-*` arms, where it is part
                # of the architecture under test.
                nn.BatchNorm2d(cur),
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
    g = Generator(cfg.z_dim, equalized_lr=cfg.equalized_lr, pixel_norm=cfg.pixel_norm,
                  match_out_scale=cfg.match_out_scale)
    d = Discriminator(equalized_lr=cfg.equalized_lr, minibatch_std=cfg.minibatch_std)
    return g, d
