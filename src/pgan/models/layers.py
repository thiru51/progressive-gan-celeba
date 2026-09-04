"""The four pieces ProGAN adds on top of a DCGAN.

Each one is a separate class so the ablation can switch it off without touching
the rest of the network. That is the whole point of this repository: the paper
(Karras et al., ICLR 2018) presents progressive growing, equalised learning
rates, pixel normalisation and the minibatch-stddev layer as a package, and does
not report what any single one contributes on its own.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedConv2d(nn.Module):
    """Conv2d whose weights are scaled by He's constant at every forward pass.

    Standard practice is to bake the He constant into the initial weights and
    then leave them alone. ProGAN instead initialises from N(0, 1) and multiplies
    by the constant on the fly. The difference only shows up with an adaptive
    optimiser: Adam normalises each parameter's update by its own gradient
    magnitude, so a layer initialised with small weights takes proportionally
    huge steps early on. Applying the scale after the optimiser sees the weight
    puts every layer on the same effective learning rate.
    """

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, bias=True, gain=math.sqrt(2)):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale = gain / math.sqrt(in_ch * kernel * kernel)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, gain=math.sqrt(2)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale = gain / math.sqrt(in_dim)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)


class EqualizedConvTranspose2d(nn.Module):
    """The transposed-convolution counterpart of EqualizedConv2d.

    The fan-in is the trap. ConvTranspose2d stores its weight as
    (in_channels, out_channels, kh, kw) -- the opposite layout from Conv2d -- so
    the number of inputs feeding each output unit is out_channels * kh * kw.
    Using shape[0] here instead of shape[1] scales every generator layer wrong by
    a constant, which shows up as a learning rate that will not tune rather than
    as an obvious error.
    """

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, bias=True, gain=math.sqrt(2)):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(in_ch, out_ch, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale = gain / math.sqrt(out_ch * kernel * kernel)

    def forward(self, x):
        return F.conv_transpose2d(
            x, self.weight * self.scale, self.bias, self.stride, self.padding
        )


class PixelNorm(nn.Module):
    """Normalise each pixel's feature vector to unit length.

    Applied in the generator only. It has no learnable parameters, so unlike
    batch norm it cannot amplify anything -- it just stops G and D from entering
    a magnitude escalation, where G answers a stronger discriminator by inflating
    activations rather than changing what it draws.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)


class MinibatchStdDev(nn.Module):
    """Append one channel holding the average per-feature stddev of the batch.

    This is the only place the discriminator sees more than one sample at a time.
    A generator that collapses to a single mode produces batches with near-zero
    spread, which lands in a constant feature map the discriminator can read
    directly, so mode collapse becomes cheap to detect and therefore expensive
    for G to commit to.

    `group_size` follows the paper: the batch is split into groups of 4 and the
    statistic is computed within each, which keeps it meaningful at large batch
    sizes. Batches that do not divide evenly fall back to one single group.
    """

    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        n, c, h, w = x.shape
        g = self.group_size if n % self.group_size == 0 else n
        y = x.view(g, n // g, c, h, w)
        y = y - y.mean(dim=0, keepdim=True)
        y = y.pow(2).mean(dim=0)
        y = torch.sqrt(y + 1e-8)
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(g, 1, h, w)
        return torch.cat([x, y], dim=1)


class Blur(nn.Module):
    """Binomial low-pass filter used in place of a bare nearest-neighbour resize.

    Not one of the four headline contributions, but the reference implementation
    uses it and dropping it leaves visible checkerboard artefacts, so it is here
    and is switched by the same flag as the rest of the ProGAN stack.
    """

    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = (k[:, None] * k[None, :])
        k = k / k.sum()
        self.register_buffer("kernel", k.expand(channels, 1, 3, 3).contiguous())
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=self.channels)
