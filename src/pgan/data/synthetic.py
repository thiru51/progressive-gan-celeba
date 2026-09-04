"""A tiny procedural stand-in for CelebA.

Its only job is to let the whole pipeline -- training loop, growing schedule,
FID, checkpointing -- be tested end to end without a 1.4 GB download. It is
never a substitute for the real measurement, and nothing produced from it is
reported anywhere.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class Blobs(Dataset):
    """Coloured ellipses on a gradient. Cheap, but has structure to learn."""

    def __init__(self, n=4096, size=64, seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:size, 0:size] / size
        imgs = np.empty((n, size, size, 3), dtype=np.uint8)
        for i in range(n):
            cx, cy = rng.uniform(0.3, 0.7, 2)
            rx, ry = rng.uniform(0.12, 0.30, 2)
            colour = rng.uniform(0.2, 1.0, 3)
            bg = rng.uniform(0.0, 0.4, 3)
            m = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
            img = bg[None, None, :] + (yy * 0.3)[:, :, None]
            img = np.where(m[:, :, None], colour[None, None, :], img)
            imgs[i] = np.clip(img * 255, 0, 255).astype(np.uint8)
        self.arr = imgs

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        x = torch.from_numpy(self.arr[i]).permute(2, 0, 1).float()
        return x.div_(127.5).sub_(1.0)
