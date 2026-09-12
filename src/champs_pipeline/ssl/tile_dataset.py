#!/usr/bin/env python3
"""Map-style dataset over the CHAMPS tile shards, for DINOv2 training.

The shards are plain WebDataset-layout tars of JPEG tiles (see
``scripts/slides/ssl/ssl_shards.py``). DINOv2's training harness uses map-style datasets
with its own samplers/collate, so rather than an iterable WebDataset we index
every JPEG's byte offset once and random-access it: ``tarfile`` exposes each
member's ``offset_data`` (where the file bytes start in the tar), so at read
time we just ``seek`` in the raw tar file and read ``size`` bytes — no tar
parsing, fast random access, and a clean drop-in for DINOv2's ImageNet-style
loader.

Build the index once (cached next to the shards); training then loads it
instantly. One raw file handle is cached per shard per worker.
"""
from __future__ import annotations

import glob
import io
import os
import random
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

INDEX_NAME = "shard_index.npz"


def build_shard_index(shard_dir: str | Path, out_path: str | Path | None = None,
                      holdout_slides: set | None = None) -> str:
    """Scan every tar once and record (shard_id, offset_data, size) per JPEG.

    ``holdout_slides`` (slide_ids) are skipped so the SSL training index excludes
    the in-domain probe holdout (the encoder must not see probe slides).
    """
    shard_dir = Path(shard_dir)
    tars = sorted(glob.glob(str(shard_dir / "*.tar")))
    if not tars:
        raise FileNotFoundError(f"no *.tar shards under {shard_dir}")
    shard_paths: list[str] = []
    rows: list[tuple[int, int, int]] = []
    n_held = 0
    n_trunc = 0
    for si, tp in enumerate(tars):
        shard_paths.append(os.path.abspath(tp))
        try:
            with tarfile.open(tp, "r") as t:
                for m in t:
                    if m.isfile() and m.name.endswith(".jpg"):
                        if holdout_slides and m.name.rsplit(".", 2)[0] in holdout_slides:
                            n_held += 1
                            continue
                        rows.append((si, m.offset_data, m.size))
        except tarfile.ReadError:
            # truncated tar (leftover from a cancelled cut) -> keep the valid
            # members read before the truncation point, skip the rest.
            n_trunc += 1
    if n_trunc:
        print(f"  tolerated {n_trunc} truncated tar(s)")
    if holdout_slides:
        print(f"  excluded {n_held:,} tiles from {len(holdout_slides)} holdout slides")
    entries = np.asarray(rows, dtype=np.int64)
    out_path = str(out_path or (shard_dir / INDEX_NAME))
    np.savez(out_path, entries=entries, shards=np.array(shard_paths))
    # np.savez appends .npz if missing
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    print(f"indexed {len(entries):,} tiles across {len(shard_paths)} shards -> {out_path}")
    return out_path


class ChampsTileDataset(Dataset):
    """Random-access dataset of JPEG tiles across the shard tars."""

    def __init__(self, shard_dir: str | Path | None = None, transform=None,
                 index_path: str | Path | None = None, *, root: str | Path | None = None,
                 target_transform=None, holdout_file: str | Path | None = None,
                 extra: str | Path | None = None):
        # ``root``/``extra`` are dinov2's generic dataset kwargs (make_dataset); map
        # root->shard_dir and extra->holdout_file so it is config-drivable via
        # dataset_path "ChampsTiles:root=<shards>:extra=<holdout.txt>".
        shard_dir = Path(shard_dir if shard_dir is not None else root)
        holdout_file = holdout_file or extra
        self.target_transform = target_transform
        # A holdout list (in-domain probe slides) -> a separate training index that
        # excludes them, so the encoder never trains on probe data.
        holdout = None
        if holdout_file and os.path.exists(str(holdout_file)):
            holdout = {l.strip() for l in open(holdout_file) if l.strip()}
            index_path = index_path or (shard_dir / "shard_index_train.npz")
        index_path = str(index_path or (shard_dir / INDEX_NAME))
        if not os.path.exists(index_path):
            index_path = build_shard_index(shard_dir, index_path, holdout_slides=holdout)
        z = np.load(index_path, allow_pickle=False)
        self.entries = z["entries"]           # (N, 3): shard_id, offset_data, size
        self.shards = [str(s) for s in z["shards"]]
        self.transform = transform
        self._fh: dict[int, object] = {}      # per-shard raw file handle (per worker)

    def __len__(self) -> int:
        return len(self.entries)

    def _handle(self, shard_id: int):
        fh = self._fh.get(shard_id)
        if fh is None:
            fh = open(self.shards[shard_id], "rb", buffering=0)
            self._fh[shard_id] = fh
        return fh

    def __getitem__(self, i: int):
        # A few tiles at the truncation point of leftover tars are partial and
        # won't decode; substitute another random tile rather than crash the run.
        for _ in range(16):
            shard_id, offset, size = (int(v) for v in self.entries[i])
            try:
                fh = self._handle(shard_id)
                fh.seek(offset)
                data = fh.read(size)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img, 0  # DINOv2 ignores the label for SSL
            except Exception:
                i = random.randrange(len(self.entries))
        raise RuntimeError("too many undecodable tiles in a row")

    def __getstate__(self):
        # don't pickle open file handles across worker fork/spawn
        s = self.__dict__.copy()
        s["_fh"] = {}
        return s

