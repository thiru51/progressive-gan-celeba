"""ProGAN generator and discriminator, with growing as a switchable behaviour.

The reason growing is a flag rather than a separate class: the paper bundles
progressive growing with three other changes, so comparing "DCGAN" against
"ProGAN" says nothing about which part mattered. Here the same weights, the same
channel schedule and the same blocks can be trained either by fading in one
resolution at a time or straight at the target resolution, and the two runs
differ in nothing else.

Resolution schedule for 64x64 is 4 -> 8 -> 16 -> 32 -> 64, five stages.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pgan.models.layers import (
    EqualizedConv2d,
    EqualizedConvTranspose2d,
    EqualizedLinear,
    MinibatchStdDev,
    PixelNorm,
)

RESOLUTIONS = [4, 8, 16, 32, 64]
# Width per resolution. Capped at 512 at the low end because a 4x4 map with more
# channels than that costs parameters without adding spatial detail.
WIDTHS = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64}


class _GBlock(nn.Module):
    """Upsample then two 3x3 convolutions, the generator's per-resolution unit."""

    def __init__(self, in_ch, out_ch, pixel_norm=True):
        super().__init__()
        self.c1 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        self.c2 = EqualizedConv2d(out_ch, out_ch, 3, padding=1)
        self.norm = PixelNorm() if pixel_norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # Nearest-neighbour upsampling rather than a transposed convolution: a
        # stride-2 transposed conv with an even kernel puts a checkerboard into
        # every output, and the paper avoids it for exactly that reason.
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.norm(self.act(self.c1(x)))
        return self.norm(self.act(self.c2(x)))


class Generator(nn.Module):
    def __init__(self, z_dim=128, img_ch=3, max_res=64, pixel_norm=True):
        super().__init__()
        self.z_dim = z_dim
        self.max_res = max_res
        self.stages = [r for r in RESOLUTIONS if r <= max_res]

        self.norm = PixelNorm() if pixel_norm else nn.Identity()
        act = nn.LeakyReLU(0.2, inplace=True)
        w0 = WIDTHS[4]
        # Pixel norm goes after *every* activation, not once at the end of the
        # block. An earlier version normalised only the block output, which left
        # the 3x3 convolution reading unnormalised activations -- a quiet
        # deviation from the reference that is invisible in the output shapes.
        self.initial = nn.Sequential(
            EqualizedConvTranspose2d(z_dim, w0, 4, 1, 0),
            act,
            PixelNorm() if pixel_norm else nn.Identity(),
            EqualizedConv2d(w0, w0, 3, padding=1),
            act,
        )

        self.blocks = nn.ModuleList()
        for prev, cur in zip(self.stages, self.stages[1:]):
            self.blocks.append(_GBlock(WIDTHS[prev], WIDTHS[cur], pixel_norm))

        # One toRGB per resolution. During a fade the old one is still needed, so
        # they cannot be replaced in place as the network grows.
        self.to_rgb = nn.ModuleList([
            EqualizedConv2d(WIDTHS[r], img_ch, 1, gain=1.0) for r in self.stages
        ])

    def forward(self, z, stage=None, alpha=1.0):
        """stage indexes RESOLUTIONS; alpha in [0,1] fades the newest block in."""
        stage = len(self.stages) - 1 if stage is None else stage
        if z.dim() == 2:
            z = z[:, :, None, None]

        h = self.norm(self.initial(self.norm(z)))
        if stage == 0:
            return torch.tanh(self.to_rgb[0](h))

        for block in self.blocks[:stage - 1]:
            h = block(h)
        prev = h
        h = self.blocks[stage - 1](h)

        out = self.to_rgb[stage](h)
        if alpha < 1.0:
            # The skip path is the previous resolution's RGB, upsampled. At
            # alpha=0 the new block contributes nothing, so growing does not
            # discard what the shallower network already learned.
            skip = F.interpolate(self.to_rgb[stage - 1](prev), scale_factor=2, mode="nearest")
            out = alpha * out + (1.0 - alpha) * skip
        return torch.tanh(out)


class _DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = EqualizedConv2d(in_ch, in_ch, 3, padding=1)
        self.c2 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        return F.avg_pool2d(x, 2)


class Discriminator(nn.Module):
    def __init__(self, img_ch=3, max_res=64, minibatch_std=True):
        super().__init__()
        self.max_res = max_res
        self.stages = [r for r in RESOLUTIONS if r <= max_res]
        act = nn.LeakyReLU(0.2, inplace=True)

        self.from_rgb = nn.ModuleList([
            EqualizedConv2d(img_ch, WIDTHS[r], 1) for r in self.stages
        ])
        # Mirrors the generator: blocks[i] maps stage i+1 down to stage i.
        self.blocks = nn.ModuleList()
        for prev, cur in zip(self.stages, self.stages[1:]):
            self.blocks.append(_DBlock(WIDTHS[cur], WIDTHS[prev]))

        self.mbstd = MinibatchStdDev() if minibatch_std else None
        w0 = WIDTHS[4] + (1 if minibatch_std else 0)
        self.final = nn.Sequential(
            EqualizedConv2d(w0, WIDTHS[4], 3, padding=1),
            act,
            EqualizedConv2d(WIDTHS[4], WIDTHS[4], 4),
            act,
        )
        self.out = EqualizedLinear(WIDTHS[4], 1, gain=1.0)

    def forward(self, x, stage=None, alpha=1.0):
        stage = len(self.stages) - 1 if stage is None else stage

        h = F.leaky_relu(self.from_rgb[stage](x), 0.2)
        if stage > 0:
            h = self.blocks[stage - 1](h)
            if alpha < 1.0:
                # Downsample the input and read it through the previous
                # resolution's fromRGB, so D also sees the old path during a fade.
                skip = F.leaky_relu(
                    self.from_rgb[stage - 1](F.avg_pool2d(x, 2)), 0.2
                )
                h = alpha * h + (1.0 - alpha) * skip
            for block in reversed(self.blocks[:stage - 1]):
                h = block(h)

        if self.mbstd is not None:
            h = self.mbstd(h)
        return self.out(self.final(h).flatten(1)).squeeze(1)


def build(cfg):
    g = Generator(cfg.z_dim, max_res=cfg.resolution, pixel_norm=cfg.pixel_norm)
    d = Discriminator(max_res=cfg.resolution, minibatch_std=cfg.minibatch_std)
    return g, d
