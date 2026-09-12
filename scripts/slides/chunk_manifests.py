"""Write the manifests the extraction jobs read, one per chunk of slides.

Columns: ``wsi`` (path relative to the data root), ``patient_id`` (case),
``notes`` (dataset name). No resolution column: Trident reads magnification
and resolution from the file. With ``--coords-encoder``, each row also
carries ``coords_path``, the tissue coordinate file of that encoder, so the
features are extracted on its tiling (used for the in-domain encoder).
"""

import argparse
import os
import re

import pandas as pd

from champs_pipeline.data_prep.feature_index import load_feature_index, usable_files


def coords_path(feature_path):
    """The coordinate file that belongs to a feature file."""
    return re.sub(r"features_[^/]+/(.+)\.h5$", r"patches/\1_patches.h5", feature_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--site", default=None, help="Restrict to one site.")
    ap.add_argument("--slides", default=None, help="CSV with a slide_id column; restrict to these.")
    ap.add_argument("--coords-encoder", default=None,
                    help="Reuse this encoder's coordinate files (needs --index).")
    ap.add_argument("--index", default=None, help="Feature index, for --coords-encoder.")
    ap.add_argument("--name", default=None, help="Dataset name (default: the site, or 'all').")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-size", type=int, default=100)
    args = ap.parse_args()

    inventory = pd.read_csv(args.inventory, low_memory=False)
    # Only slides OpenSlide could open.
    inventory = inventory[inventory["probe_error"].isna()]
    if args.site:
        inventory = inventory[inventory["site"] == args.site]
    if args.slides:
        wanted = set(pd.read_csv(args.slides)["slide_id"].astype(str))
        inventory = inventory[inventory["slide_id"].isin(wanted)]

    name = args.name or (args.site.lower() if args.site else "all")
    # The manifest columns Trident reads; the case is the patient id.
    manifest = pd.DataFrame({
        "wsi": inventory["wsi_path"].values,
        "patient_id": inventory["case_id"].values,
        "notes": name,
    })
    if args.coords_encoder:
        index = load_feature_index(args.index)
        files = usable_files(index, args.coords_encoder).set_index("slide_id")["path"]
        feature_paths = inventory["slide_id"].astype(str).map(files)
        manifest = manifest[feature_paths.notna().values]
        manifest["coords_path"] = [coords_path(p) for p in feature_paths.dropna()]

    os.makedirs(args.out_dir, exist_ok=True)
    n_chunks = 0
    for start in range(0, len(manifest), args.chunk_size):
        chunk = manifest.iloc[start:start + args.chunk_size]
        chunk.to_csv(os.path.join(args.out_dir, f"{name}_chunk_{n_chunks}.csv"), index=False)
        n_chunks += 1
    print(f"{len(manifest)} slides in {n_chunks} chunks under {args.out_dir}")


if __name__ == "__main__":
    main()
