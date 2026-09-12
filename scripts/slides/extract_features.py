"""Segment tissue, tile, and extract tile features with Trident.

One invocation processes one chunk manifest (column ``wsi``, paths relative
to ``--wsi-dir``) into one job directory:

    <job-dir>/20x_<tile>px_0px_overlap/patches/<slide>_patches.h5
    <job-dir>/20x_<tile>px_0px_overlap/features_<encoder>/<slide>.h5

Magnification and resolution are read from each slide file. ``--task feat``
extracts features on coordinate files already in the job directory; a
manifest with a ``coords_path`` column has them copied there first, which
is how the in-domain encoder runs on the tiling of a public encoder.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch

from champs_pipeline.encoders import trident_adaptations

# Tile size in pixels at the target magnification, per encoder.
TILE_SIZE = {
    "virchow2": 224,
    "hoptimus0": 224,
    "hoptimus1": 224,
    "champs_mits": 224,
    "uni_v2": 256,
    "conch_v15": 512,
}
MAGNIFICATION = 20
SEGMENTER = "hest"
SEG_CONF_THRESH = 0.5


def copy_coordinate_files(manifest, coords_dir):
    """Place the manifest's coordinate files where Trident's feature task expects them."""
    patches_dir = coords_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    for wsi, source in zip(manifest["wsi"], manifest["coords_path"]):
        target = patches_dir / f"{Path(wsi).stem}_patches.h5"
        if not target.exists():
            shutil.copy2(source, target)


def run_segmentation(processor, batch_size, device):
    from trident.segmentation_models.load import segmentation_model_factory

    segmenter = segmentation_model_factory(SEGMENTER, confidence_thresh=SEG_CONF_THRESH)
    processor.run_segmentation_job(
        segmenter,
        seg_mag=segmenter.target_mag,
        holes_are_tissue=True,
        batch_size=batch_size,
        device=device,
    )


def run_patching(processor, tile_size):
    processor.run_patching_job(target_magnification=MAGNIFICATION, patch_size=tile_size, overlap=0)


def run_feature_extraction(processor, encoder_name, weights, coords_dir_name, batch_size, device):
    from trident.patch_encoder_models.load import encoder_factory

    encoder = encoder_factory(encoder_name, weights_path=weights)
    processor.run_patch_feature_extraction_job(
        coords_dir=coords_dir_name,
        patch_encoder=encoder,
        device=device,
        saveas="h5",
        batch_limit=batch_size,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", required=True, choices=sorted(TILE_SIZE))
    ap.add_argument("--weights", default=None, help="Local checkpoint; required for champs_mits.")
    ap.add_argument("--wsi-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--job-dir", type=Path, required=True)
    ap.add_argument("--task", default="all", choices=["all", "seg", "coords", "feat"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    # Tile size in pixels at the target magnification, per encoder.
    tile_size = TILE_SIZE[args.encoder]
    # The directory name Trident expects for coordinate files.
    coords_dir_name = f"{MAGNIFICATION}x_{tile_size}px_0px_overlap"
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    manifest = pd.read_csv(args.manifest)
    if "coords_path" in manifest.columns:
        copy_coordinate_files(manifest, args.job_dir / coords_dir_name)

    trident_adaptations.apply()
    from trident import Processor

    processor = Processor(
        job_dir=str(args.job_dir),
        wsi_source=args.wsi_dir,
        custom_list_of_wsis=args.manifest,
        max_workers=args.max_workers,
    )
    tasks = ["seg", "coords", "feat"] if args.task == "all" else [args.task]
    for task in tasks:
        if task == "seg":
            run_segmentation(processor, args.batch_size, device)
        elif task == "coords":
            run_patching(processor, tile_size)
        else:
            run_feature_extraction(processor, args.encoder, args.weights, coords_dir_name,
                                   args.batch_size, device)


if __name__ == "__main__":
    main()
