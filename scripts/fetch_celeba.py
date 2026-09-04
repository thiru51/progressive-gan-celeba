#!/usr/bin/env python
"""Download the CelebA aligned-face shards.

The canonical CelebA link is a Google Drive folder that rate-limits and breaks
torchvision's downloader most days. The HuggingFace mirror below is the same
202,599 aligned JPEGs, served over plain HTTPS with resumable range requests, so
this is the source the repo standardises on.

Nothing here is unpacked -- `python -m pgan.data.prepare` does the decoding.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "nielsr/CelebA-faces"
SHARDS = [
    "data/train-00000-of-00003.parquet",
    "data/train-00001-of-00003.parquet",
    "data/train-00002-of-00003.parquet",
]
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def human(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def download(url, dest: Path, expect=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expect is None or dest.stat().st_size == expect):
        print(f"  have {dest.name} ({human(dest.stat().st_size)})")
        return dest

    # Resume a partial file rather than restarting a 460 MB download.
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  resuming {dest.name} from {human(have)}")

    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        done = have
        with open(dest, mode) as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                pct = 100 * done / total if total else 0
                print(f"\r  {dest.name}  {human(done)} / {human(total)}  {pct:5.1f}%",
                      end="", flush=True)
    print()
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw", help="where the parquet shards land")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be fetched, and its size, without fetching")
    a = p.parse_args()

    out = Path(a.out)
    if a.dry_run:
        import json
        api = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
        with urllib.request.urlopen(api, timeout=30) as r:
            meta = json.load(r)
        total = 0
        for f in meta:
            size = f.get("size") or (f.get("lfs") or {}).get("size") or 0
            total += size
            print(f"  {f['path']:40s} {human(size)}")
        print(f"  {'TOTAL':40s} {human(total)}   -> {out.resolve()}")
        return 0

    print(f"source: huggingface.co/datasets/{REPO}")
    for s in SHARDS:
        download(BASE + s, out / Path(s).name)
    print(f"\ndone. next: python -m pgan.data.prepare --raw {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
