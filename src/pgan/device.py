"""Device and precision selection, kept in one place so runs are comparable."""
from __future__ import annotations

import contextlib

import torch


def resolve(pref=None):
    if pref and pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup(device):
    if device.type != "cuda":
        return
    # TF32 on the matmul path costs nothing visible at this scale and buys ~1.3x.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def amp_dtype(device, want_amp=True):
    """bf16 where supported, else fp16, else nothing.

    bf16 is preferred over fp16 despite the smaller mantissa: GAN training runs
    without a loss scaler here, and fp16 gradients underflow to zero in the
    discriminator's early layers long before bf16 loses anything that matters.
    """
    if not want_amp or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast(device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def describe(device):
    if device.type != "cuda":
        return "cpu"
    p = torch.cuda.get_device_properties(device)
    return f"{p.name}, {p.total_memory / 1024**2:.0f} MiB, sm_{p.major}{p.minor}, torch {torch.__version__}"
