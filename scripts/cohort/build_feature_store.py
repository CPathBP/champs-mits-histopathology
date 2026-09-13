"""Build the feature store of one encoder over the training slides of the cohort.

Reads the tile-feature file of every training slide the encoder covers,
in cohort order, into one Lance dataset (features flattened to float32,
the embedding dimension, the tile count, the tile coordinates), and
writes ``lance_index_<encoder>.csv`` with the dataset path and row of
every slide. The manifests join that index. An existing dataset is
never overwritten.
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import lance
import numpy as np
import pandas as pd
import pyarrow as pa

SCHEMA = pa.schema([
    pa.field("slide_id", pa.string(), nullable=False),
    pa.field("features", pa.large_list(pa.float32()), nullable=False),
    pa.field("embed_dim", pa.int32(), nullable=False),
    pa.field("num_patches", pa.int32(), nullable=False),
    pa.field("coords_x", pa.list_(pa.int32()), nullable=False),
    pa.field("coords_y", pa.list_(pa.int32()), nullable=False),
])


def read_features(item):
    """``(slide_id, features, coords)`` of one feature file; refuses an empty or bad one."""
    slide_id, path = item
    with h5py.File(path, "r") as handle:
        features = np.asarray(handle["features"], dtype=np.float32)
        coords = np.asarray(handle["coords"], dtype=np.int32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise RuntimeError(f"{path}: features have shape {features.shape}")
    if coords.shape != (features.shape[0], 2):
        raise RuntimeError(f"{path}: coords {coords.shape} do not match features {features.shape}")
    return slide_id, features, coords


def record_batch(rows):
    return pa.RecordBatch.from_arrays([
        pa.array([r[0] for r in rows], type=pa.string()),
        pa.array([r[1].reshape(-1) for r in rows], type=pa.large_list(pa.float32())),
        pa.array([r[1].shape[1] for r in rows], type=pa.int32()),
        pa.array([r[1].shape[0] for r in rows], type=pa.int32()),
        pa.array([r[2][:, 0] for r in rows], type=pa.list_(pa.int32())),
        pa.array([r[2][:, 1] for r in rows], type=pa.list_(pa.int32())),
    ], schema=SCHEMA)


def batches(items, batch_size, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(items), batch_size):
            rows = list(pool.map(read_features, items[start:start + batch_size]))
            yield record_batch(rows)
            print(f"converted {min(start + batch_size, len(items))}/{len(items)}", flush=True)


def validate(dataset_path, items):
    """Row count, slide order and three round trips against the source files."""
    dataset = lance.dataset(dataset_path)
    if dataset.count_rows() != len(items):
        raise RuntimeError(f"{dataset.count_rows()} rows written for {len(items)} slides")
    written = dataset.to_table(columns=["slide_id"])["slide_id"].to_pylist()
    if written != [slide_id for slide_id, _ in items]:
        raise RuntimeError("slide order differs from the cohort order")
    samples = sorted({0, len(items) // 2, len(items) - 1})
    table = dataset.take(samples, columns=["features", "embed_dim", "coords_x", "coords_y"])
    for position, source in enumerate(samples):
        _, features, coords = read_features(items[source])
        dim = int(table["embed_dim"][position].as_py())
        got = np.asarray(table["features"][position].values, dtype=np.float32).reshape(-1, dim)
        np.testing.assert_array_equal(got, features)
        got_x = np.asarray(table["coords_x"][position].values, dtype=np.int32)
        got_y = np.asarray(table["coords_y"][position].values, dtype=np.int32)
        np.testing.assert_array_equal(np.stack([got_x, got_y], axis=1), coords)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-slides", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort_slides, low_memory=False, dtype={"slide_id": str})
    column = f"feature_path_{args.encoder}"
    rows = cohort[cohort["training"] & cohort[column].notna()]
    items = [(slide_id, Path(path)) for slide_id, path in zip(rows["slide_id"], rows[column])]
    absent = [path for _, path in items if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} feature files missing; first: {absent[0]}")
    dataset_path = args.out_dir / f"{args.encoder}.lance"
    if dataset_path.exists():
        raise FileExistsError(f"{dataset_path} exists; remove it to rebuild")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    building = args.out_dir / f".{args.encoder}.building"
    if building.exists():
        shutil.rmtree(building)
    print(f"{len(items)} slides -> {dataset_path}", flush=True)
    try:
        reader = pa.RecordBatchReader.from_batches(SCHEMA, batches(items, args.batch_size,
                                                                   args.workers))
        lance.write_dataset(reader, building, schema=SCHEMA, mode="create")
        validate(building, items)
        building.rename(dataset_path)
    except BaseException:
        if building.exists():
            shutil.rmtree(building)
        raise
    index = pd.DataFrame({"slide_id": [s for s, _ in items],
                          "lance_dataset_path": str(dataset_path.absolute()),
                          "lance_row_idx": range(len(items))})
    index.to_csv(args.out_dir / f"lance_index_{args.encoder}.csv", index=False)
    print(f"wrote {dataset_path} and its row index")


if __name__ == "__main__":
    main()
