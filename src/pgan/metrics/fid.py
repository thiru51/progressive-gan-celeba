"""Frechet Inception Distance.

Read this before quoting any number out of this repository.

The features come from **torchvision's** ImageNet InceptionV3, not from the
ported TF-Slim graph (`pt_inception-2015-12-05`) that `pytorch-fid` and every
published FID table use. The two networks have different weights, so the two
scales are different: a FID of 30 here is not a FID of 30 in the ProGAN paper,
and the numbers in RESULTS.md must never be compared against published ones.

Within this repository the comparison is valid, because every arm is scored by
this same function against the same held-out reference images with the same
sample count. That is what the ablation needs. Absolute realism is not what is
being measured.

FID is also biased by sample count -- fewer samples give a systematically higher
score -- so the count is recorded alongside every number.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

FEATURE_DIM = 2048
# ImageNet statistics; torchvision's InceptionV3 was trained with these.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class InceptionFeatures(torch.nn.Module):
    """InceptionV3 truncated at the 2048-d average pool."""

    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                           transform_input=False, aux_logits=True)
        # The classifier head is what we are cutting off; dropout would also make
        # the features stochastic, which would put noise straight into the mean
        # and covariance.
        net.fc = torch.nn.Identity()
        net.dropout = torch.nn.Identity()
        net.eval()
        self.net = net
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_uint8):
        """x_uint8: (N, 3, H, W) uint8 in 0..255."""
        x = x_uint8.float().div_(255.0)
        # antialias matters: 64 -> 299 is an upsample, but the same helper is used
        # on 178px reference crops elsewhere, and silent aliasing there would
        # shift the real-image statistics only.
        x = F.interpolate(x, size=(299, 299), mode="bilinear",
                          align_corners=False, antialias=True)
        x = (x - _MEAN.to(x.device)) / _STD.to(x.device)
        return self.net(x)


def _stats(feats: np.ndarray):
    mu = feats.mean(axis=0)
    # rowvar=False: each row is a sample, each column a feature.
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """||mu1-mu2||^2 + Tr(S1 + S2 - 2 sqrt(S1 S2))."""
    from scipy import linalg

    diff = mu1 - mu2
    # scipy deprecated `disp` in 1.16 and drops it in 1.18; both spellings are
    # accepted here so the metric does not change value with the scipy version.
    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    except Exception:
        covmean = linalg.sqrtm(sigma1.dot(sigma2) + np.eye(sigma1.shape[0]) * eps)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        # sqrtm of a product of two near-singular covariances can fall off the
        # real axis entirely. Nudging the diagonal is the standard repair and is
        # what the reference implementation does.
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        imag = np.max(np.abs(covmean.imag))
        if imag > 1e-3:
            raise ValueError(f"sqrtm returned a complex matrix (max imag {imag:.3g})")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


@torch.no_grad()
def features_from_batches(model, batches, device, total=None):
    out = []
    seen = 0
    for b in batches:
        b = b.to(device, non_blocking=True)
        out.append(model(b).float().cpu().numpy())
        seen += b.shape[0]
        if total and seen >= total:
            break
    feats = np.concatenate(out, axis=0)
    return feats[:total] if total else feats


def fid_from_features(real_feats, fake_feats):
    m1, s1 = _stats(real_feats)
    m2, s2 = _stats(fake_feats)
    return frechet_distance(m1, s1, m2, s2)
