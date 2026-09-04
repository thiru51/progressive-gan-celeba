"""The ProGAN layers, checked against what they are supposed to do numerically."""
import math

import pytest
import torch

from pgan.models.layers import (
    Blur,
    EqualizedConv2d,
    EqualizedConvTranspose2d,
    EqualizedLinear,
    MinibatchStdDev,
    PixelNorm,
)


def test_pixelnorm_gives_unit_length_feature_vectors():
    x = torch.randn(4, 32, 8, 8) * 7.0
    y = PixelNorm()(x)
    # Every spatial position's channel vector should have RMS 1.
    rms = y.pow(2).mean(dim=1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_pixelnorm_is_scale_invariant():
    x = torch.randn(2, 16, 4, 4)
    n = PixelNorm()
    assert torch.allclose(n(x), n(x * 100.0), atol=1e-4)


def test_pixelnorm_survives_an_all_zero_pixel():
    x = torch.zeros(1, 8, 2, 2)
    assert torch.isfinite(PixelNorm()(x)).all()


def test_equalized_conv_scale_matches_he_constant():
    c = EqualizedConv2d(8, 16, 3)
    assert c.scale == pytest.approx(math.sqrt(2) / math.sqrt(8 * 3 * 3))


def test_equalized_transpose_uses_out_channels_as_fan_in():
    """ConvTranspose2d stores (in, out, kh, kw), so the fan-in is out*kh*kw.

    Reading it off shape[0] instead is the easiest way to get this wrong, and it
    scales every generator layer by a constant that looks like a bad LR.
    """
    c = EqualizedConvTranspose2d(4, 16, 3)
    assert c.weight.shape == (4, 16, 3, 3)
    assert c.scale == pytest.approx(math.sqrt(2) / math.sqrt(16 * 3 * 3))


def test_equalized_layers_preserve_activation_scale():
    # A He-scaled layer on unit-variance input should come out near unit variance.
    x = torch.randn(256, 64, 8, 8)
    y = EqualizedConv2d(64, 64, 3, padding=1, bias=False)(x)
    assert 0.7 < y.std().item() < 1.6


def test_equalized_linear_shapes():
    y = EqualizedLinear(10, 3)(torch.randn(5, 10))
    assert y.shape == (5, 3)


def test_minibatch_stddev_adds_exactly_one_channel():
    x = torch.randn(8, 16, 4, 4)
    y = MinibatchStdDev()(x)
    assert y.shape == (8, 17, 4, 4)
    assert torch.allclose(y[:, :16], x)


def test_minibatch_stddev_channel_is_constant_across_space():
    y = MinibatchStdDev()(torch.randn(8, 16, 4, 4))
    ch = y[:, -1]
    assert torch.allclose(ch, ch[:, :1, :1].expand_as(ch), atol=1e-6)


def test_minibatch_stddev_reads_zero_on_a_collapsed_batch():
    """The whole point of the layer: identical samples produce no spread."""
    x = torch.randn(1, 16, 4, 4).repeat(8, 1, 1, 1)
    y = MinibatchStdDev()(x)
    assert y[:, -1].abs().max().item() < 1e-3


def test_minibatch_stddev_reads_higher_on_a_diverse_batch():
    same = MinibatchStdDev()(torch.randn(1, 16, 4, 4).repeat(8, 1, 1, 1))[:, -1].mean()
    varied = MinibatchStdDev()(torch.randn(8, 16, 4, 4))[:, -1].mean()
    assert varied > same + 0.1


def test_minibatch_stddev_handles_a_batch_not_divisible_by_the_group():
    y = MinibatchStdDev(group_size=4)(torch.randn(6, 8, 4, 4))
    assert y.shape == (6, 9, 4, 4) and torch.isfinite(y).all()


def test_blur_preserves_shape_and_is_normalised():
    b = Blur(3)
    assert b.kernel.sum().item() == pytest.approx(3.0)  # 1.0 per channel
    x = torch.ones(2, 3, 8, 8)
    y = b(x)
    # Interior pixels of a constant image must come back unchanged.
    assert torch.allclose(y[:, :, 1:-1, 1:-1], x[:, :, 1:-1, 1:-1], atol=1e-5)
