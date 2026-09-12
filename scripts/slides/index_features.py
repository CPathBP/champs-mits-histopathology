"""Index every tile-feature file under the feature store.

One row per ``features_<encoder>/<slide>.h5``: ``slide_id``, ``encoder``,
``resolution`` (the parent directory, for example ``20x_224px_0px_overlap``),
``path``, ``timestamp``, the tiling attributes read from the file
(``level0_magnification``, ``patch_size_level0``, ``n_patches``), and from
the inventory the slide's ``wsi_path``, ``site``, ``case_id``,
``id_conflict``, and ``wsi_objective_power``. ``refused`` names why a file is
left out of every downstream table: unreadable, empty, no scanner power, or
a recorded magnification that differs from the scanner's; it is empty
otherwise. With ``--study-id-mapping``, ``champs_deid`` is joined on
``case_id``.
"""

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

# Index column -> inventory column.
INVENTORY_COLUMNS = {
    "wsi_path": "wsi_path",
    "site": "site",
    "case_id": "case_id",
    "id_conflict": "id_conflict",
    "wsi_objective_power": "scanner_power",
}


def describe_file(path):
    """The index row of one feature file, from its path and modification time."""
    parts = path.split("/")
    modified = datetime.fromtimestamp(os.path.getmtime(path))
    return {
        "slide_id": os.path.splitext(parts[-1])[0],
        "encoder": parts[-2][len("features_"):],
        "resolution": parts[-3],
        "path": os.path.abspath(path),
        "timestamp": modified.strftime("%Y-%m-%d %H:%M:%S"),
    }


def read_tiling(path):
    """Magnification, tile size at level 0, and tile count; NaN and -1 when unreadable."""
    import h5py

    try:
        with h5py.File(path, "r") as h5:
            attrs = h5["coords"].attrs
            magnification = float(attrs.get("level0_magnification", "nan"))
            tile_size = float(attrs.get("patch_size_level0", "nan"))
            n_patches = int(h5["features"].shape[0])
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan"), -1
    return magnification, tile_size, n_patches


def refusal_reason(index):
    """Why a file is left out, or None; later rules override earlier ones."""
    reason = pd.Series(None, index=index.index, dtype=object)
    reason[index["wsi_objective_power"].isna()] = "no scanner power"
    mismatch = index["level0_magnification"] != index["wsi_objective_power"]
    reason[mismatch] = "magnification differs from scanner"
    reason[index["n_patches"] == 0] = "empty"
    reason[index["n_patches"] < 0] = "unreadable"
    return reason


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--study-id-mapping", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pattern = os.path.join(args.features_root, "**", "features_*", "*.h5")
    files = sorted(glob.glob(pattern, recursive=True))
    index = pd.DataFrame([describe_file(f) for f in files])
    with ThreadPoolExecutor(args.workers) as pool:
        tiling = list(pool.map(read_tiling, index["path"]))
    index[["level0_magnification", "patch_size_level0", "n_patches"]] = tiling

    # A slide id that names several files in the store resolves to the first by path.
    inventory = pd.read_csv(args.inventory, low_memory=False)
    inventory = inventory.sort_values("wsi_path").drop_duplicates("slide_id").set_index("slide_id")
    for column, source in INVENTORY_COLUMNS.items():
        index[column] = index["slide_id"].map(inventory[source])
    index["refused"] = refusal_reason(index)

    if args.study_id_mapping:
        mapping = pd.read_csv(args.study_id_mapping).rename(columns={"study_id": "case_id"})
        index = index.merge(mapping, on="case_id", how="left")

    index = index.sort_values(["encoder", "site", "timestamp"], ascending=[True, True, False])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    index.to_csv(args.out, index=False)

    per_encoder = index.drop_duplicates(["encoder", "slide_id"]).groupby("encoder").size()
    refused = index[index["refused"].notna()].groupby(["encoder", "refused"]).size()
    print(f"{len(index)} feature files; slides per encoder:\n{per_encoder.to_string()}")
    print(f"refused files:\n{refused.to_string() if len(refused) else 'none'}")


if __name__ == "__main__":
    main()
