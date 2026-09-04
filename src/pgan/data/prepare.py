"""Turn the CelebA parquet shards into one memory-mapped uint8 array.

202,599 images as individual files means 202,599 file opens per epoch, and on a
spinning-rust or network filesystem that alone dominates the step time. A single
(N, 64, 64, 3) uint8 array is 2.5 GB, memory-maps in constant time and lets the
OS page cache do the work.

Crop geometry follows the DCGAN/ProGAN convention for aligned CelebA: take the
central 148x148 of the 178x218 aligned image, then resize to the target. Faces
in the aligned set are registered on eye position, so this crop lands the face
consistently and drops most of the background.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CROP = 148
SRC_SIZE = (178, 218)


def center_crop_resize(img: Image.Image, size: int) -> np.ndarray:
    w, h = img.size
    left = (w - CROP) // 2
    top = (h - CROP) // 2
    img = img.convert("RGB").crop((left, top, left + CROP, top + CROP))
    # BICUBIC rather than the default: downsampling 148 -> 64 with nearest or
    # bilinear leaves aliasing that a discriminator learns to key on, which
    # flatters the model for the wrong reason.
    img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def iter_images(raw: Path):
    import pyarrow.parquet as pq

    shards = sorted(raw.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no parquet shards in {raw}. Run: python scripts/fetch_celeba.py"
        )
    for shard in shards:
        pf = pq.ParquetFile(shard)
        col = "image" if "image" in pf.schema_arrow.names else pf.schema_arrow.names[0]
        # Row groups, not the whole shard: a 460 MB shard decoded at once peaks
        # over 4 GB of RSS.
        for batch in pf.iter_batches(batch_size=512, columns=[col]):
            for cell in batch.column(0).to_pylist():
                data = cell["bytes"] if isinstance(cell, dict) else cell
                yield Image.open(io.BytesIO(data))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/celeba64.npy")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="stop after N images (for testing)")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Two passes would mean decoding 200k JPEGs twice, so instead grow a list of
    # chunks and concatenate once. Peak memory is the final array, 2.5 GB.
    chunks, buf = [], []
    n = 0
    for img in iter_images(Path(a.raw)):
        buf.append(center_crop_resize(img, a.size))
        n += 1
        if len(buf) == 8192:
            chunks.append(np.stack(buf))
            buf.clear()
            print(f"\r  decoded {n:,}", end="", flush=True)
        if a.limit and n >= a.limit:
            break
    if buf:
        chunks.append(np.stack(buf))
    print(f"\r  decoded {n:,}")

    arr = np.concatenate(chunks, axis=0)
    np.save(out, arr)

    meta = {
        "count": int(arr.shape[0]),
        "size": a.size,
        "crop": CROP,
        "dtype": str(arr.dtype),
        "layout": "NHWC uint8, 0-255",
        "source": "huggingface nielsr/CelebA-faces (aligned)",
        "bytes": int(arr.nbytes),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  {arr.shape}  {arr.nbytes / 1e9:.2f} GB")
    print(f"wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
