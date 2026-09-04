"""Dataset over the prepared memmap, plus the train / FID-reference split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Images held out of training and used only as the real side of FID. Comparing
# generated samples against the same images the generator was trained on rewards
# memorisation; this split is what makes the number mean "looks like faces"
# rather than "reproduces these faces".
HOLDOUT = 20_000


class CelebA64(Dataset):
    def __init__(self, path="data/celeba64.npy", split="train", holdout=HOLDOUT):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run scripts/fetch_celeba.py then "
                f"python -m pgan.data.prepare"
            )
        # mmap_mode keeps the 2.5 GB out of process memory; workers share the
        # OS page cache instead of each holding a copy.
        self.arr = np.load(self.path, mmap_mode="r")
        n = len(self.arr)
        if holdout >= n:
            raise ValueError(f"holdout {holdout} >= dataset size {n}")
        self.split = split
        self.lo, self.hi = (0, n - holdout) if split == "train" else (n - holdout, n)

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, i):
        # uint8 HWC 0..255 -> float CHW -1..1, matching the generator's tanh.
        # np.array, not asarray: the memmap slice is read-only and torch warns
        # (correctly) about wrapping non-writable memory in a tensor.
        x = np.array(self.arr[self.lo + i], dtype=np.uint8)
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        return x.div_(127.5).sub_(1.0)

    @property
    def meta(self):
        j = self.path.with_suffix(".json")
        return json.loads(j.read_text()) if j.exists() else {}


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Inverse of the normalisation above, for saving grids and for FID.

    The round() is not cosmetic. Casting a float to uint8 truncates, so a pixel
    that normalises to 0.9999 comes back as 0 rather than 1 -- an off-by-one on
    roughly half of all values, applied to the real images as well as the
    generated ones, which biases the reference statistics FID is measured
    against. test_uint8_round_trip_is_lossless pins this.
    """
    return x.add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)


def infinite(loader):
    """Cycle a DataLoader forever.

    Training is counted in gradient steps, not epochs, so that every arm sees
    exactly the same number of updates regardless of how the growing schedule
    splits them up.
    """
    while True:
        yield from loader
