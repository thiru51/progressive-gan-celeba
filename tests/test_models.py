"""Shapes, growing behaviour and the flags that define each ablation arm."""
import pytest
import torch

from pgan.config import ARMS, make
from pgan.models import dcgan, progan
from pgan.models.layers import EqualizedConv2d, MinibatchStdDev, PixelNorm
from pgan.train import build_models


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_builds_and_round_trips(arm):
    cfg = make(arm)
    g, d = build_models(cfg)
    z = torch.randn(2, cfg.z_dim)
    x = g(z)
    assert x.shape == (2, 3, cfg.resolution, cfg.resolution)
    assert torch.isfinite(x).all()
    assert d(x).shape == (2,)


def test_generator_output_is_inside_tanh_range():
    x = dcgan.Generator()(torch.randn(4, 128))
    assert x.min() >= -1.0 and x.max() <= 1.0


@pytest.mark.parametrize("stage,res", list(enumerate(progan.RESOLUTIONS)))
def test_progan_emits_every_resolution(stage, res):
    g, d = progan.Generator(), progan.Discriminator()
    x = g(torch.randn(2, 128), stage=stage, alpha=1.0)
    assert x.shape == (2, 3, res, res)
    assert d(x, stage=stage, alpha=1.0).shape == (2,)


def test_alpha_zero_falls_back_to_the_previous_resolution():
    """At alpha=0 the new block must contribute nothing.

    That is what makes growing safe: the moment a stage is added the network
    still produces exactly what it produced before, only upsampled.
    """
    g = progan.Generator().eval()
    z = torch.randn(2, 128)
    with torch.no_grad():
        prev = g(z, stage=1, alpha=1.0)
        faded = g(z, stage=2, alpha=0.0)
        # atanh both sides: the blend happens before the tanh, so comparing the
        # post-tanh images directly would fail on a nonlinearity, not on logic.
        upsampled = torch.nn.functional.interpolate(
            prev.clamp(-0.999, 0.999).atanh(), scale_factor=2, mode="nearest")
    assert torch.allclose(faded.clamp(-0.999, 0.999).atanh(), upsampled, atol=1e-4)


def test_alpha_moves_the_output_continuously():
    g = progan.Generator().eval()
    z = torch.randn(2, 128)
    with torch.no_grad():
        outs = [g(z, stage=3, alpha=a) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    diffs = [(b - a).abs().mean().item() for a, b in zip(outs, outs[1:])]
    assert all(d > 0 for d in diffs), "alpha had no effect somewhere"
    assert max(diffs) < 10 * min(diffs), f"discontinuous fade: {diffs}"


def test_flags_actually_change_the_module_tree():
    plain_g = dcgan.Generator(equalized_lr=False, pixel_norm=False)
    pn_g = dcgan.Generator(equalized_lr=False, pixel_norm=True)
    assert not any(isinstance(m, PixelNorm) for m in plain_g.modules())
    assert any(isinstance(m, PixelNorm) for m in pn_g.modules())
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in plain_g.modules())
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in pn_g.modules())

    eq_g = dcgan.Generator(equalized_lr=True)
    assert not any(isinstance(m, torch.nn.ConvTranspose2d) for m in eq_g.modules())

    assert not any(isinstance(m, MinibatchStdDev)
                   for m in dcgan.Discriminator(minibatch_std=False).modules())
    assert any(isinstance(m, MinibatchStdDev)
               for m in dcgan.Discriminator(minibatch_std=True).modules())
    assert any(isinstance(m, EqualizedConv2d)
               for m in dcgan.Discriminator(equalized_lr=True).modules())


def test_the_two_progan_arms_are_architecturally_identical():
    """progan-fixed and progan-grow must differ only in the schedule.

    If their parameter shapes ever diverge, the growing comparison is measuring
    two different networks and is worthless.
    """
    fixed = dict((k, tuple(v.shape)) for k, v in build_models(make("progan-fixed"))[0].state_dict().items())
    grow = dict((k, tuple(v.shape)) for k, v in build_models(make("progan-grow"))[0].state_dict().items())
    assert fixed == grow


def test_unknown_arm_is_rejected():
    with pytest.raises(KeyError):
        make("not-an-arm")


def test_grow_requires_the_progan_architecture():
    with pytest.raises(ValueError):
        make("dcgan", grow=True)
