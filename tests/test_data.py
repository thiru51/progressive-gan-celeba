"""The data path: normalisation, the holdout split, and the crop geometry."""
import numpy as np
import pytest
import torch

from pgan.data.celeba import CelebA64, to_uint8
from pgan.data.prepare import CROP, center_crop_resize
from pgan.data.synthetic import Blobs


@pytest.fixture
def fake_dataset(tmp_path):
    """A 200-image stand-in written in the same layout prepare.py produces."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(200, 64, 64, 3), dtype=np.uint8)
    p = tmp_path / "celeba64.npy"
    np.save(p, arr)
    return p, arr


def test_normalisation_lands_in_tanh_range(fake_dataset):
    p, _ = fake_dataset
    ds = CelebA64(p, holdout=50)
    x = ds[0]
    assert x.shape == (3, 64, 64)
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_uint8_round_trip_is_lossless(fake_dataset):
    p, arr = fake_dataset
    ds = CelebA64(p, holdout=50)
    back = to_uint8(ds[7]).permute(1, 2, 0).numpy()
    assert np.array_equal(back, arr[7])


def test_train_and_holdout_do_not_overlap(fake_dataset):
    p, _ = fake_dataset
    tr = CelebA64(p, split="train", holdout=50)
    ho = CelebA64(p, split="holdout", holdout=50)
    assert len(tr) == 150 and len(ho) == 50
    assert tr.hi <= ho.lo, "training and FID reference images overlap"


def test_holdout_returns_the_tail_of_the_array(fake_dataset):
    p, arr = fake_dataset
    ho = CelebA64(p, split="holdout", holdout=50)
    assert np.array_equal(to_uint8(ho[0]).permute(1, 2, 0).numpy(), arr[150])


def test_a_holdout_larger_than_the_dataset_is_rejected(fake_dataset):
    p, _ = fake_dataset
    with pytest.raises(ValueError):
        CelebA64(p, holdout=500)


def test_missing_data_says_what_to_run():
    with pytest.raises(FileNotFoundError, match="prepare"):
        CelebA64("does/not/exist.npy")


def test_crop_is_centred_and_the_right_size():
    from PIL import Image
    # A 178x218 image with a marked centre, the real aligned-CelebA geometry.
    a = np.zeros((218, 178, 3), dtype=np.uint8)
    a[218 // 2, 178 // 2] = [255, 0, 0]
    out = center_crop_resize(Image.fromarray(a), 64)
    assert out.shape == (64, 64, 3)
    assert out[:, :, 0].argmax() > 0, "the marked centre pixel was cropped away"


def test_crop_size_matches_the_documented_constant():
    assert CROP == 148


def test_synthetic_stand_in_is_shaped_like_the_real_thing():
    ds = Blobs(n=8, size=64)
    x = ds[0]
    assert x.shape == (3, 64, 64) and -1.0 <= x.min() and x.max() <= 1.0
