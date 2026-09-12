# Pipeline

This document lists the steps from the primary data to the manuscript, in
the order they run. Each step is one script under `scripts/`, takes
explicit paths, and writes named files. Steps marked *GPU* need one GPU;
steps marked *per chunk* run once per chunk manifest, which is a natural
job array.

Paths below: `DATA_ROOT` is the slide store, `FEATURES_ROOT` the feature
store, `ART` the directory of the small artifacts.

## 1. Slides

From the whole-slide image store to tile features, the feature index, the
stain classifier, and the in-domain encoder.

| Step | Command | Output |
|---|---|---|
| Inventory | `python scripts/slides/crawl_wsi.py --data-root $DATA_ROOT --out $ART/wsi_inventory.csv` | one row per slide file with the study id of its case directory, the study id in its name when there is one, whether the two disagree, and the scanner's objective power and resolution |
| Chunk manifests | `python scripts/slides/chunk_manifests.py --inventory $ART/wsi_inventory.csv --site <Site> --out-dir $ART/chunks` | `<site>_chunk_<N>.csv`, 100 slides each, no resolution column |
| Features, *GPU*, *per chunk* | `python scripts/slides/extract_features.py --encoder <name> --wsi-dir $DATA_ROOT --manifest <chunk.csv> --job-dir $FEATURES_ROOT/<name>/<site>/chunk_<N>` | tissue coordinates and tile features per slide |
| Feature index | `python scripts/slides/index_features.py --features-root $FEATURES_ROOT --inventory $ART/wsi_inventory.csv --study-id-mapping $DATA_ROOT/champs_deid_studyid_mapping.csv --out $ART/feature_index.csv` | one row per feature file with its tiling attributes, the slide's scanner power, and `refused` naming why a file is left out (unreadable, empty, no scanner power, magnification differs from the scanner's) |
| In-domain features, *GPU*, *per chunk* | `python scripts/slides/chunk_manifests.py --inventory $ART/wsi_inventory.csv --index $ART/feature_index.csv --coords-encoder hoptimus0 --slides <cohort.csv> --name indomain --out-dir $ART/chunks`, then `extract_features.py --encoder champs_mits --weights <checkpoint> --task feat ...` | features of the in-domain encoder on the tissue coordinates of the H-optimus-0 run |
| Stain classifier | `python scripts/slides/train_stain_classifier.py --index $ART/feature_index.csv --cache $ART/meanpool_virchow2.npz --out $ART/stain_classifier.pkl --pred-csv $ART/stain_classifier_oof.csv` | classifier and its out-of-fold predictions |
| Stain scores | `python scripts/slides/score_stains.py --index $ART/feature_index.csv --model $ART/stain_classifier.pkl --cache $ART/meanpool_virchow2.npz --out $ART/stain_scores.csv` | `p_non_HE` per slide; the cohort construction drops slides at or above 0.5 |
| Warm start | `python scripts/slides/ssl/warmstart.py --out $ART/hoptimus0_dinov2_warmstart.pth` | H-optimus-0 as a DINOv2 checkpoint |
| SSL manifest | `python scripts/slides/ssl/ssl_manifest.py --index $ART/feature_index.csv --data-root $DATA_ROOT --out $ART/ssl_manifest.csv` | slides and coordinate files of the training corpus |
| SSL shards, *per job* | `python scripts/slides/ssl/ssl_shards.py --manifest $ART/ssl_manifest.csv --out-dir $ART/ssl_shards --shard-id <J> --num-jobs <N>` | tar shards of JPEG tiles, 200 per slide |
| SSL training, *GPU* | `torchrun --nproc_per_node=8 <dinov2>/train/train.py --config-file configs/ssl/champs_vitg14_blockexp.yaml --output-dir <run>` with `scripts/slides/ssl` on `PYTHONPATH` | the in-domain encoder (teacher backbone of the final checkpoint) |

Settings as run: segmentation with the `hest` segmenter at confidence 0.5,
holes treated as tissue; tiles at 20x with 224 px (Virchow2, H-optimus-0,
H-optimus-1, in-domain), 256 px (UNI2-h), 512 px (CONCH v1.5); no overlap.
Magnification and resolution are read from each slide file. A slide's case is
its directory; a file whose name carries a different study id is flagged
(`id_conflict`) and excluded by the cohort construction. Consumers of
the feature index take, per slide and encoder, the newest file the index
does not refuse (`champs_pipeline.data_prep.feature_index`). Trident is installed unmodified; `champs_pipeline.encoders.trident_adaptations`
registers the in-domain encoder and satisfies the timm version check of the
H-optimus loaders.

The later parts (reports, cohort and reference labels, training, inference,
evaluation, reader study, manuscript) are added here as they are released.
