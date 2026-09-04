"""FID, checked against cases where the answer is known in closed form."""
import numpy as np
import pytest

from pgan.metrics.fid import fid_from_features, frechet_distance


def test_identical_features_give_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 32))
    assert fid_from_features(a, a) == pytest.approx(0.0, abs=1e-6)


def test_a_pure_mean_shift_gives_the_squared_distance():
    """With equal covariances, FID collapses to ||mu1 - mu2||^2."""
    d, shift = 32, 2.0
    mu1, mu2 = np.zeros(d), np.full(d, shift)
    sigma = np.eye(d)
    assert frechet_distance(mu1, sigma, mu2, sigma) == pytest.approx(d * shift ** 2)


def test_a_pure_scale_change_gives_the_trace_term():
    """N(0, I) vs N(0, 4I): Tr(I + 4I - 2*sqrt(4I)) = d * (1 + 4 - 4) = d."""
    d = 32
    mu = np.zeros(d)
    assert frechet_distance(mu, np.eye(d), mu, 4 * np.eye(d)) == pytest.approx(d, rel=1e-5)


def test_it_is_symmetric():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(400, 16)), rng.normal(loc=0.5, size=(400, 16))
    assert fid_from_features(a, b) == pytest.approx(fid_from_features(b, a), rel=1e-6)


def test_it_grows_with_the_gap():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(800, 16))
    scores = [fid_from_features(a, rng.normal(loc=s, size=(800, 16)))
              for s in (0.0, 0.5, 1.0, 2.0)]
    assert scores == sorted(scores)


def test_same_distribution_scores_near_zero_but_not_at_it():
    """Finite-sample FID is biased upward. Worth pinning so nobody reads a small
    positive number as evidence of a difference."""
    rng = np.random.default_rng(3)
    s = fid_from_features(rng.normal(size=(2000, 64)), rng.normal(size=(2000, 64)))
    assert 0.0 < s < 5.0


def test_singular_covariances_do_not_produce_nan():
    """Real feature matrices are rank-deficient whenever n < 2048, which is the
    normal case for a quick evaluation."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=(20, 64))
    b = rng.normal(size=(20, 64))
    assert np.isfinite(fid_from_features(a, b))
