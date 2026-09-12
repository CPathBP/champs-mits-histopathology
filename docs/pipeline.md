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

## 2. Reports

From the pathology report export to the extracted findings and the case
attribution. `REPORTS` is the diagnosis table of the report export.

| Step | Command | Output |
|---|---|---|
| Corpus | `python scripts/reports/build_corpus.py --reports-csv $REPORTS --out-dir $ART/corpus --n-shards 32 --seed 42` | `corpus_cases.txt` (every case whose report carries H&E text after HTML stripping), 32 shard lists dealt round-robin after a seeded shuffle, `manifest.json` |
| Extraction, *GPU*, *per shard* | `python scripts/reports/extract_findings.py --reports-csv $REPORTS --schema configs/extraction/morphology_schema_v4_5.yaml --prompt configs/extraction/morphology_v4_5.j2 --case-list <shard.txt> --out-dir $ART/extraction --base-url http://<host>:<port>/v1 --model <served name>` | `per_case/<case>.json` with the findings, negations, quality flags, uncertain records and diagnosis lines of the case, each with a supporting quote; `run_meta.json` with the schema and prompt fingerprints |
| Findings table | `python scripts/reports/finalize_extraction.py --per-case $ART/extraction/per_case --reports-csv $REPORTS --schema <schema> --prompt <prompt> --corpus $ART/corpus/corpus_cases.txt --out-dir $ART` | `findings.parquet` (one row per case, organ, text source, kind and finding, with severity, extent, modifiers and quotes; site-report records copied into the CPL review where it concurred, flagged as such; records replicated between paired organs flagged), `case_status.csv` (every corpus case: extracted, failed, or missing), `case_attribution.parquet` (reporting template and text origin per case), `extraction_record.json` (the inference settings, counts, every corrected value, the share of verbatim quotes and of quotes inside the section the model named) |
| Validation against annotations | `python scripts/reports/score_against_annotations.py --findings $ART/findings.parquet --annotations <annotation directory> --out-dir $ART/validation` | per finding, precision and recall of the extraction against a reader's annotation of the report text over the reviewed cells, with 95% intervals, severity agreement on the four-bin ladder, abstentions, and the share of cases without any error |
| Corpus statistics | `python scripts/reports/describe_corpus.py --findings $ART/findings.parquet --attribution $ART/case_attribution.parquet --corpus $ART/corpus/corpus_cases.txt --out-dir $ART/corpus_statistics` | records per organ, text source and kind; qualifier values per finding; cases per reporting template and text origin; the model's text source against the model-free origin; per organ and source, the corpus cases with a record, a condition, a negation, a quality flag only, or nothing |

Settings as run: the extraction sends one chat completion per case to a
vLLM server (an OpenAI-compatible endpoint) serving Gemma 4 31B IT in NVFP4
with an fp8 KV cache and a context of 40,960 tokens, at temperature 0 with
at most 8,192 output tokens. The prompt renders the schema and the case's
cleaned H&E text; when a case has several report rows, the row with the
longest text represents it. The reply is constrained to the JSON schema
derived from the extraction schema (organs, text sources, finding names,
and the unions of the severity, extent and modifier values), so every
record names a schema value. When the records are flattened, a record
whose organ the finding does not allow is dropped, a severity, extent or
modifier value outside the finding's own lists is nulled, and an uncertain
record that shares its quote with a positive masks that positive; every
such correction is counted in the run record. A case whose reply cannot be
parsed is recorded as failed. Quantised inference is not
bit-reproducible, so a re-run reproduces the records statistically, not
exactly (`env/EXTERNALS.md`).

## 3. Cohort, Reference A, manifests, folds

From the slide arm, the report arm and the release tables to what training
reads. `L2` is the directory of the release tables, `MAP` the study-id
mapping, `STORE` the directory of the feature stores.

| Step | Command | Output |
|---|---|---|
| Cohort | `python scripts/cohort/build_cohort.py --inventory $ART/wsi_inventory.csv --index $ART/feature_index.csv --stain-scores $ART/stain_scores.csv --study-id-mapping $MAP --findings $ART/findings.parquet --attribution $ART/case_attribution.parquet --encoders configs/cohort/encoders.yaml --demographics $L2/CHAMPS_deid_basic_demographics.csv --decode $L2/CHAMPS_deid_decode_results.csv --out-dir $ART/cohort` | `cohort_slides.csv` (every H&E slide of a lung or the liver with its case, slide source, inclusion flags and per-encoder feature paths), `cohort_cases.csv` (every case of the report table with its inclusion flags, demographics and cause-of-death group), `dropped_slides.csv` (every drop with its stage and reason), `funnel.json` (the box counts of the report arm and the slide arm) |
| Reference A | `python scripts/cohort/build_reference_a.py --findings $ART/findings.parquet --schema configs/extraction/morphology_schema_v4_5.yaml --cohort-dir $ART/cohort --screen configs/cohort/screen_cases.csv --config configs/cohort/lung.yaml --config configs/cohort/liver.yaml --out-dir $ART/reference_a` | `reference_a_case.parquet` (per case, organ, slide source and finding: what the central laboratory's description asserts), `reference_a_slide.parquet` (per training slide, finding and variant: the label, the keep mask and the reason for a mask) |
| Feature store, *per encoder* | `python scripts/cohort/build_feature_store.py --cohort-slides $ART/cohort/cohort_slides.csv --encoder virchow2 --out-dir $STORE` | `<encoder>.lance` and `lance_index_<encoder>.csv` (the dataset row of every training slide) |
| Manifests | `python scripts/cohort/build_manifests.py --cohort-dir $ART/cohort --reference-dir $ART/reference_a --findings $ART/findings.parquet --schema <schema> --config configs/cohort/lung.yaml --config configs/cohort/liver.yaml --encoders configs/cohort/encoders.yaml --lance-dir $STORE --out-dir $ART/manifests` | `manifest_<organ>_<variant>.csv`, one row per training slide with the label, mask, severity, extent, modifier and flag columns, the feature paths and store rows of every encoder, and the demographics |
| Folds | `python scripts/cohort/build_folds.py --manifest-dir $ART/manifests --config configs/cohort/lung.yaml --config configs/cohort/liver.yaml --encoders configs/cohort/encoders.yaml --out-dir $ART/folds --seed 42` | `case_splits.csv` (organ, design, fold, case, split), `fold_csvs/<organ>_<variant>_<design>/fold_<k>.csv` (the manifest with a split column), `gate_table.csv` and `gate_report.md` (class counts per fold and label with a pass, warn or fail status) |
| Tables | `python scripts/cohort/render_funnel.py --cohort-dir $ART/cohort --out-dir $ART/tables`; `python scripts/cohort/render_table1.py --cohort-dir $ART/cohort --out-dir $ART/tables`; `python scripts/cohort/render_finding_burden.py --reference-dir $ART/reference_a --cohort-dir $ART/cohort --decode-map configs/cohort/decode_histology_map.csv --config configs/cohort/lung.yaml --config configs/cohort/liver.yaml --out-dir $ART/tables` | the funnel tables and figure, Table 1 (full and compact), the finding supply and the candidate table, each a read of the cohort artifacts |

Rules as run. A slide enters the linked cohort when its name carries no
stain token and a lung or liver tissue code, its study id maps to a case
with an examined report, and some report section describes that organ for
that slide source (the site report and the central laboratory's review of
the site scans describe site-scanned slides; the central laboratory's own
slide descriptions describe its own scans). A linked slide enters the
training cohort when the description that labels it is the central
laboratory's (the site report is not a label source), it carries a finding
or a negation and not only quality flags, it is not flagged inadequate for
diagnosis, the stain classifier gives a non-H&E probability below 0.5, and a correctly scaled feature file
exists for the cohort encoder. Every drop is one row of `dropped_slides.csv`.

Reference A states, per case, organ, slide source and finding, `positive`
when the description records the finding, `uncertain` when it only hedges
it, and silence otherwise. The `raw` label variant keeps silence as a
negative and masks the uncertain cells. The `elig` variant also masks a
negative whose case records or hedges the finding in any section of that
organ, whose description names a related finding that makes the silence
uninformative (diffuse alveolar damage or fibrin lining the alveoli for
hyaline membranes, unspecified pigment for hemozoin pigment), or whose
description is out of focus or severely autolysed. The `elig_screen`
variant adds the direct-test screen: a negative whose case the screen
lists for that finding (`configs/cohort/screen_cases.csv`) is masked as
well. The screen is a frozen input: case and finding pairs, for
bronchopneumonia and pneumonitis in the lungs and hemozoin pigment in the
liver, whose case has a positive direct test (tissue PCR panel,
immunohistochemistry, PCR, special stain or clinical laboratory result)
while the report does not record the finding; it was drawn from an
earlier label analysis that is not part of this repository, and the file
names the evidence channel and trigger of every pair. A positive is never
masked in any variant. The fold designs
share one balanced assignment of cases to ten microfolds (site, per-finding
case positives and their interaction, slide count; seed 42): five outer
folds of two test microfolds and one validation microfold; one fold per
site with a validation tenth of the other sites' cases; and the five-fold
design over the slides every compared encoder covers.

The later parts (training, inference, evaluation, reader study,
manuscript) are added here as they are released.
