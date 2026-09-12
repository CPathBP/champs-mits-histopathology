"""List the slides and coordinate files the in-domain encoder trains on.

One row per slide with a tissue coordinate file at 20x and 256 px tiles:
the UNI2-h extraction covers nearly every slide, and any other encoder with
that tiling fills the rest. No stain filter. Columns: ``slide_id``,
``wsi_path``, ``coords_h5``, ``location``.
"""

import argparse
import os
import re

import pandas as pd

from champs_pipeline.data_prep.feature_index import load_feature_index

RESOLUTION = "20x_256px_0px_overlap"
PREFERRED_ENCODER = "uni_v2"


def coords_path(feature_path):
    return re.sub(r"features_[^/]+/(.+)\.h5$", r"patches/\1_patches.h5", feature_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = load_feature_index(args.index)
    candidates = index[
        (index["resolution"] == RESOLUTION) & index["wsi_path"].notna() & index["refused"].isna()
    ].copy()
    candidates["preferred"] = candidates["encoder"] == PREFERRED_ENCODER
    candidates = candidates.sort_values(["preferred", "encoder", "timestamp"],
                                        ascending=[False, True, False])
    chosen = candidates.drop_duplicates("slide_id")

    manifest = pd.DataFrame({
        "slide_id": chosen["slide_id"],
        "wsi_path": [os.path.join(args.data_root, p) for p in chosen["wsi_path"]],
        "coords_h5": [coords_path(p) for p in chosen["path"]],
        "location": chosen["site"],
    }).sort_values("slide_id")
    manifest.to_csv(args.out, index=False)
    print(f"{len(manifest)} slides; by encoder {chosen['encoder'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
