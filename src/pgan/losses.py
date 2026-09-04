"""Adversarial losses.

Both arms of the ablation use the same loss. That is a deliberate deviation from
the papers -- DCGAN used the non-saturating logistic loss, ProGAN used WGAN-GP --
because changing the objective at the same time as the architecture makes the
comparison unreadable. WGAN-GP is implemented here so the deviation can be
checked rather than argued about.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def d_loss_ns(real_logits, fake_logits):
    """Standard non-saturating discriminator loss, written in softplus form.

    softplus(-x) is log(1 + e^-x) = -log(sigmoid(x)) computed without ever
    forming sigmoid(x), so a confident discriminator does not produce inf.
    """
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def g_loss_ns(fake_logits):
    # -log D(G(z)) rather than log(1 - D(G(z))): the latter has vanishing
    # gradient exactly where G is losing, which is when it needs one most.
    return F.softplus(-fake_logits).mean()


def d_loss_wgan(real_logits, fake_logits):
    return fake_logits.mean() - real_logits.mean()


def g_loss_wgan(fake_logits):
    return -fake_logits.mean()


def gradient_penalty(discriminator, real, fake, forward_kwargs=None):
    """WGAN-GP: penalise ||grad|| deviating from 1 on the real-fake interpolate."""
    kw = forward_kwargs or {}
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device, dtype=real.dtype)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    logits = discriminator(mixed, **kw)
    grad = torch.autograd.grad(
        outputs=logits.sum(), inputs=mixed, create_graph=True, only_inputs=True
    )[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def r1_penalty(discriminator, real, forward_kwargs=None):
    """R1: penalise the gradient on real samples only (Mescheder et al., 2018).

    Cheaper than WGAN-GP -- one backward instead of an interpolation plus a
    backward -- and it is the regulariser that made the later StyleGAN line
    stable, so it is the more useful knob to expose.
    """
    kw = forward_kwargs or {}
    real = real.detach().requires_grad_(True)
    logits = discriminator(real, **kw)
    grad = torch.autograd.grad(
        outputs=logits.sum(), inputs=real, create_graph=True, only_inputs=True
    )[0]
    return grad.flatten(1).pow(2).sum(1).mean()
