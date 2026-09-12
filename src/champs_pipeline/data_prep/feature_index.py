"""Loading and file selection for the feature index (``scripts/slides/index_features.py``)."""

import pandas as pd

COLUMNS = ("slide_id", "encoder", "resolution", "path", "timestamp", "level0_magnification",
           "patch_size_level0", "n_patches", "wsi_path", "wsi_objective_power", "refused")


def load_feature_index(path):
    index = pd.read_csv(path, low_memory=False)
    missing = [c for c in COLUMNS if c not in index.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    if index["path"].duplicated().any():
        raise ValueError(f"{path} lists a feature file twice")
    index["slide_id"] = index["slide_id"].astype(str)
    return index


def usable_files(index, encoder, resolution=None):
    """The newest file per slide that the index does not refuse."""
    rows = index[(index["encoder"] == encoder) & index["refused"].isna()]
    if resolution is not None:
        rows = rows[rows["resolution"] == resolution]
    return rows.sort_values("timestamp").drop_duplicates("slide_id", keep="last")


def check_coverage(index, slide_ids, encoder, resolution=None):
    """Raise unless every slide has a usable file for the encoder."""
    covered = set(usable_files(index, encoder, resolution)["slide_id"])
    missing = sorted(set(map(str, slide_ids)) - covered)
    if missing:
        raise ValueError(f"{len(missing)} slides have no usable {encoder} file; "
                         f"first: {missing[:5]}")
