#!/usr/bin/env python3
"""Cut pixel tiles into tar shards for the in-domain encoder's training.

Tiles are cut with Trident's patcher on the coordinates of each slide's
``*_patches.h5``, so they have the geometry of the extracted features. Up to
``--tiles-per-slide`` tiles are sampled per slide with a seed derived from
the slide name; near-blank tiles are dropped. Output: tar files of
``<slide>.<index>.jpg``, read at training time by
``champs_pipeline.ssl.tile_dataset``. Job ``J`` of ``--num-jobs`` takes the
manifest rows with ``row_index % num_jobs == J``.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI


def read_coords(coords_h5: str | Path):
    """Return (coords[N,2] level-0 int, attrs dict) from a Trident patches h5."""
    with h5py.File(coords_h5, "r") as f:
        coords = f["coords"][:]
        attrs = {k: f["coords"].attrs[k] for k in f["coords"].attrs}
    return coords, attrs


def pick_indices(n: int, k: int, rng: random.Random) -> list[int]:
    """Deterministically choose up to ``k`` of ``n`` tile indices."""
    if k <= 0 or k >= n:
        return list(range(n))
    return sorted(rng.sample(range(n), k))


def informative(tile: Image.Image, min_std: float) -> bool:
    """Drop near-blank tiles (background inside the tissue contour)."""
    if min_std <= 0:
        return True
    g = np.asarray(tile.convert("L"), dtype=np.float32)
    return float(g.std()) >= min_std


def iter_slide_tiles(wsi_path: str, coords_h5: str, k: int, seed: str, min_std: float):
    """Yield (tile_index, x, y, PIL.Image) for a subsample of a slide's tiles."""
    coords, attrs = read_coords(coords_h5)
    rng = random.Random(f"{seed}:{Path(wsi_path).name}")
    idx = pick_indices(len(coords), k, rng)
    wsi = OpenSlideWSI(str(wsi_path), lazy_init=False)
    patcher = wsi.create_patcher(
        patch_size=int(attrs["patch_size"]),
        src_mag=int(attrs["level0_magnification"]),
        dst_mag=int(attrs["target_magnification"]),
        custom_coords=coords,
        coords_only=False,
        pil=True,
    )
    n = len(patcher)
    for i in idx:
        if i >= n:
            continue
        tile, x, y = patcher[i]
        if not isinstance(tile, Image.Image):
            tile = Image.fromarray(tile)
        tile = tile.convert("RGB")
        if informative(tile, min_std):
            yield i, int(x), int(y), tile


class ShardWriter:
    """Rotating WebDataset tar writer: new part every ``max_count`` samples."""

    def __init__(self, out_dir: Path, prefix: str, max_count: int, jpeg_quality: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.max_count = max_count
        self.jpeg_quality = jpeg_quality
        self.part = 0
        self.count_in_part = 0
        self.total = 0
        self.tar: tarfile.TarFile | None = None
        out_dir.mkdir(parents=True, exist_ok=True)

    def _open(self):
        path = self.out_dir / f"{self.prefix}-{self.part:05d}.tar"
        self.tar = tarfile.open(path, "w")

    def write(self, key: str, tile: Image.Image):
        if self.tar is None or self.count_in_part >= self.max_count:
            self.close()
            self._open()
            self.count_in_part = 0
        buf = io.BytesIO()
        tile.save(buf, format="JPEG", quality=self.jpeg_quality)
        data = buf.getvalue()
        info = tarfile.TarInfo(name=f"{key}.jpg")
        info.size = len(data)
        self.tar.addfile(info, io.BytesIO(data))
        self.count_in_part += 1
        self.total += 1

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.part += 1
            self.tar = None


def process_rows(rows, out_dir: Path, prefix: str, tiles_per_slide: int, shard_size: int,
                 jpeg_quality: int, min_std: float, seed: str) -> dict:
    """Cut every slide in ``rows`` into one worker's rotating tar shards."""
    writer = ShardWriter(out_dir, prefix, shard_size, jpeg_quality)
    n_slides = 0
    n_failed = 0
    for r in rows:
        sid = str(r["slide_id"])
        try:
            for i, x, y, tile in iter_slide_tiles(
                r["wsi_path"], r["coords_h5"], tiles_per_slide, seed, min_std
            ):
                writer.write(f"{sid}.{i:06d}", tile)
            n_slides += 1
        except Exception as e:  # a bad WSI must not kill the whole shard
            n_failed += 1
            print(f"[warn] {sid}: {type(e).__name__}: {e}", file=sys.stderr)
        if n_slides % 50 == 0:
            print(f"  [{prefix}] slides={n_slides} failed={n_failed} tiles={writer.total}",
                  flush=True)
    writer.close()
    return {"slides_ok": n_slides, "slides_failed": n_failed,
            "tiles": writer.total, "tar_parts": writer.part}


def _worker(payload) -> dict:
    (rows, out_dir, prefix, tiles_per_slide, shard_size, jpeg_quality, min_std, seed) = payload
    return process_rows(rows, out_dir, prefix, tiles_per_slide, shard_size,
                        jpeg_quality, min_std, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="CSV with slide_id, wsi_path, coords_h5")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tiles-per-slide", type=int, default=200, help="Max tiles sampled per slide.")
    ap.add_argument("--shard-size", type=int, default=10000, help="Samples per tar part.")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--min-std", type=float, default=8.0,
                    help="Drop tiles whose grey-level standard deviation is below this.")
    ap.add_argument("--seed", default="champs-ssl-v1")
    ap.add_argument("--shard-id", type=int, default=0, help="This job's index (SLURM array).")
    ap.add_argument("--num-jobs", type=int, default=1, help="Total array size.")
    ap.add_argument("--prefix", default="champs-ssl",
                    help="Tar name prefix; use a distinct one to add shards to an existing dir.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel slide readers within this job (uses the allocated cores).")
    args = ap.parse_args()

    rows = pd.read_csv(args.manifest).iloc[args.shard_id :: args.num_jobs].to_dict("records")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{args.prefix}-{args.shard_id:04d}"

    if args.workers <= 1:
        agg = process_rows(rows, args.out_dir, base, args.tiles_per_slide, args.shard_size,
                           args.jpeg_quality, args.min_std, args.seed)
    else:
        from concurrent.futures import ProcessPoolExecutor
        payloads = []
        for w in range(args.workers):
            sub = rows[w :: args.workers]  # disjoint round-robin split
            if not sub:
                continue
            payloads.append((sub, args.out_dir, f"{base}-w{w:02d}", args.tiles_per_slide,
                             args.shard_size, args.jpeg_quality, args.min_std, args.seed))
        agg = {"slides_ok": 0, "slides_failed": 0, "tiles": 0, "tar_parts": 0}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for s in ex.map(_worker, payloads):
                for k in agg:
                    agg[k] += s[k]

    stats = {"shard_id": args.shard_id, "num_jobs": args.num_jobs,
             "workers": args.workers, **agg}
    (args.out_dir / f"{base}_counts.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
