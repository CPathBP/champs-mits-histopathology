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
| Reference A | `python scripts/cohort/build_reference_a.py --findings $ART/findings.parquet --schema configs/extraction/morphology_schema_v4_5.yaml --cohort-dir $ART/cohort --config configs/cohort/lung.yaml --config configs/cohort/liver.yaml --out-dir $ART/reference_a` | `reference_a_case.parquet` (per case, organ, slide source and finding: what the central laboratory's description asserts), `reference_a_slide.parquet` (per training slide, finding and variant: the label, the keep mask and the reason for a mask) |
| Feature store, *per encoder* | `python scripts/cohort/build_feature_store.py --cohort-slides $ART/cohort/cohort_slides.csv --encoder virchow2 --out-dir $STORE` | `<encoder>.lance` and `lance_index_<encoder>.csv` (the dataset row of every training slide) |
| Manifests | `python scripts/cohort/build_manifests.py --cohort-dir $ART/cohort --reference-dir $ART/reference_a --findings $ART/findings.parquet --schema <schema> --config configs/cohort/lung.yaml --config configs/cohort/liver.yaml --encoders configs/cohort/encoders.yaml --lance-dir $STORE --out-dir $ART/manifests` | `manifest_<organ>_<variant>.csv`, one row per training slide with the label, mask, severity, extent, modifier and flag columns, the feature paths and store rows of every encoder, and the demographics |
| Folds | `python scripts/cohort/build_folds.py --manifest-dir $ART/manifests --config configs/cohort/lung.yaml --config configs/cohort/liver.yaml --out-dir $ART/folds --seed 42` | `case_splits.csv` (organ, design, fold, case, split), `fold_csvs/<organ>_<variant>_<design>/fold_<k>.csv` (the manifest with a split column), `gate_table.csv` and `gate_report.md` (class counts per fold and label with a pass, warn or fail status) |
| Tables | `python scripts/cohort/render_funnel.py --cohort-dir $ART/cohort --out-dir $ART/tables`; `python scripts/cohort/render_table1.py --cohort-dir $ART/cohort --out-dir $ART/tables`; `python scripts/cohort/render_finding_burden.py --reference-dir $ART/reference_a --cohort-dir $ART/cohort --decode-map configs/cohort/decode_histology_map.csv --decode $L2/CHAMPS_deid_decode_results.csv --icd-descriptions $L2/CHAMPS_icd_descriptions.csv --icd11-correspondence configs/cohort/icd11_to_icd10.csv --config configs/cohort/lung.yaml --config configs/cohort/liver.yaml --out-dir $ART/tables` | the funnel tables and figure, Table 1 (full and compact), the finding supply, the candidate table with the selection criteria, and `cause_codes.csv` (the causal-chain codes matched by each row of the cause-of-death map) |

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
hyaline membranes, unnamed pigment not stated to be non-polarizable for
hemozoin pigment), or whose description is out of focus or severely
autolysed. A positive is never masked in either variant. The fold designs
share one balanced assignment of cases to ten microfolds (site, per-finding
case positives and their interaction, slide count; seed 42): five outer
folds of two test microfolds and one validation microfold, and one fold
per site with a validation tenth of the other sites' cases.

A finding is selected for the study when at least 250 examined cases are
positive for it in Reference A, and when a cause of death that
`decode_histology_map.csv` ties to its histology, at the primary or the
contributing tier, occurs in the causal chain (underlying cause, immediate
cause, morbid conditions 1 to 8) of at least 1% of the examined cases with a
DeCoDe record. Chain codes are matched as ICD-10; ICD-11 codes are read
through `icd11_to_icd10.csv` or, failing that, through an identical
description in the release's ICD table. The renderer stops when the
selected findings differ from the study findings.

## 4. Training

From the fold manifests and the feature stores to one trained model per
run of the training matrix. `RUNS` is the directory of the run
directories. Every training run needs one GPU; MamMIL needs the kernels of
`make env-gpu`.

| Step | Command | Output |
|---|---|---|
| Matrix | `python scripts/training/expand_matrix.py --folds-dir $ART/folds/fold_csvs --out $ART/training_matrix.csv` | one row per run of `configs/training/matrix.yaml`: run id, families, organ, label variant, fold design, fold, encoder, aggregator, learning rate, training fraction, seed |
| Train, *GPU*, *per run* | `python scripts/training/train.py --matrix $ART/training_matrix.csv --run-id <run_id> --folds-dir $ART/folds/fold_csvs --runs-dir $RUNS` | `<run_id>/` with `run.json` (identity, commit, status, selected checkpoint), `config.json`, `label_mapping.json`, `data_counts.json`, `metrics.csv`, `hparams.yaml`, `checkpoints/` |
| Registry | `python scripts/training/build_run_registry.py --matrix $ART/training_matrix.csv --runs-dir $RUNS --out $ART/run_registry.csv --require-complete` | the matrix rows with their status, commit, job, epochs and selected checkpoint |

Before a run starts, the data module refuses a fold manifest whose labels
and masks contradict each other, an encoder without features for more than
5% of a split's slides, and a training or validation split without a kept
positive and a kept negative for every finding. A complete run is kept when
its matrix row and the hashes of its fold manifest and configurations are
unchanged, and refused otherwise. An attempt that did not complete, and
whose job is no longer queued, is moved under `$RUNS/.attempts/` and the run
trains again, which is what a requeued job does. A run records a dirty tree
when tracked files changed or untracked files exist under `src/`,
`scripts/`, `configs/` or `env/`.

The families: the headline models (CLAM-MB on Virchow2 features, lung and
liver, five-fold); the site-held-out models (lung and liver); the
aggregator comparison (six aggregators, lung); the encoder comparison
(CLAM-MB on five further encoders, lung); the learning curve (six training
fractions, lung and liver); and the raw label variant (lung).

Settings as run. Every slide is one bag of tile features. The heads are
the selection findings of the organ (five lung findings, two liver
findings), and a label cell whose mask is 0 enters neither the loss nor the
metrics. The loss is binary cross-entropy per finding averaged over the
kept cells of a step, with a positive weight per finding of kept negatives
over kept positives in the training split, capped at 100. Four slides per
optimizer step, 25% of the tiles of each training bag dropped at random,
AdamW with weight decay 5e-4, a linear warm-up over two epochs from 1% of
the learning rate into a cosine decay to 1e-6, gradient norm clipped at
1.0, mixed precision. Training runs at most 30 epochs and stops after five
validation epochs without a higher macro average precision over the
selection findings; the checkpoint with the highest value is the run's
model. Every run uses the fixed learning rate of its aggregator's model
configuration and seed 42. A learning-curve run keeps a nested,
prevalence-preserving fraction of the training cases; the validation and
test splits are never subsampled.

## 5. Inference

From the trained runs to one table of scores. Inference needs one GPU;
MamMIL needs the kernels of `make env-gpu`.

| Step | Command | Output |
|---|---|---|
| Preflight | `python scripts/inference/predict.py --registry $ART/run_registry.csv --matrix $ART/training_matrix.csv --runs-dir $RUNS --folds-dir $ART/folds/fold_csvs --out $ART/predictions.parquet --preflight` | no file; the checks below for every run, without a GPU |
| Predict, *GPU* | `python scripts/inference/predict.py --registry $ART/run_registry.csv --matrix $ART/training_matrix.csv --runs-dir $RUNS --folds-dir $ART/folds/fold_csvs --out $ART/predictions.parquet --device cuda` | `predictions.parquet`, one row per run, split, slide and finding: the run axes, the case, the Reference A label and mask, the probability (`score`) and the logit; `predictions.json`, the inference record: the source commit, the environment, the settings, the hash of every input and the hash of the table |
| Check | `python scripts/inference/check_predictions.py --registry $ART/run_registry.csv --matrix $ART/training_matrix.csv --runs-dir $RUNS --folds-dir $ART/folds/fold_csvs --predictions $ART/predictions.parquet` | no file; the number of runs and prediction cells when the table is complete |

Before it scores, the script refuses a source tree with changes, a run that
is not complete or was trained from a tree with changes, and a run whose
checkpoint, fold manifest, head order or feature-store rows differ from its
training record. Each model scores the validation and test slides of its
fold once, with the checkpoint that training selected: the complete bag of
each slide, float32, dropout off, one sigmoid per head. Training slides are
not scored. A masked cell gets a score and keeps its mask. The table and its
record are written to temporary files and move to the output path only when
the table is complete, so a stopped job leaves no partial table there. The
checker compares the table cell by cell with the cells of the registry, the
head mappings and the fold manifests, and compares the input hashes in the
record with the current inputs. The evaluation uses the test rows only.

## 6. Evaluation and manuscript displays

From the checked predictions, Reference A and the cohort tables to the
tables and figures of the manuscript. `RUNS` is the directory of the run
directories. No step needs a GPU.

Each display is written to `manuscript/<name>/`. The name tells whether the
display belongs to the main text (`main_`) or to the supplementary
information (`supplementary_`). A display is written with the numbers it
shows (a figure with its source data as CSV, a table as CSV) and with
`<name>.provenance.json`: the command, the commit, whether the working tree
differed from the commit, whether the display is a draft, and the SHA-256 of
every input. A figure follows the figure requirements of npj Digital
Medicine: Arimo at 8 pt, lines of at least 1 pt, a width of at most 180 mm,
PDF and PNG at 300 dpi. A figure is not written when it violates a
requirement, or when text leaves the figure, overlaps other text or covers
the data. A table is written as the body of a LaTeX `tabular` (a `longtable`
when it runs over several pages) and as CSV; the manuscript supplies the
caption and the notes, and needs the `booktabs`, `array` and `longtable`
packages.

A display reads the predictions only after it checks them against the
complete inference record of a clean commit. To render a draft from
predictions without that record, add `--draft`: the predictions must still
agree with the prediction contract, and the provenance record then shows
`"draft": true`.

Displays of the cohort:

| Display | Command | Output |
|---|---|---|
| Cohort characteristics, and the full table with the linked cohort and the slide counts | `python scripts/manuscript/main_table_cohort.py --cohort-dir $ART/cohort --finding-supply $ART/tables/finding_supply.csv --out-dir manuscript` | `main_table_cohort`, `supplementary_table_cohort` |
| Candidate findings and the selection | `python scripts/manuscript/supplementary_table_candidate_findings.py --candidate-table $ART/tables/candidate_table.csv --out-dir manuscript` | `supplementary_table_candidate_findings` |
| Reference A per finding | `python scripts/manuscript/supplementary_tables_reference_a.py --reference-slides $ART/reference_a/reference_a_slide.parquet --out-dir manuscript` | `supplementary_table_reference_a_labels`, `supplementary_table_reference_a_grades` |
| Study cohort per site | `python scripts/manuscript/supplementary_table_sites.py --cohort-dir $ART/cohort --reference-slides $ART/reference_a/reference_a_slide.parquet --out-dir manuscript` | `supplementary_table_sites` |
| Selection of cases and slides | `python scripts/manuscript/supplementary_figure_cohort_flow.py --funnel $ART/cohort/funnel.json --out-dir manuscript` | `supplementary_figure_cohort_flow` |
| Study overview | `python scripts/manuscript/main_figure_overview.py --cohort-dir $ART/cohort --reference-slides $ART/reference_a/reference_a_slide.parquet --findings $ART/findings.parquet --case-splits $ART/folds/case_splits.csv --training-configs configs/training --slide-root $DATA_ROOT --out-dir manuscript` | `main_figure_overview` |

Displays of the evaluation:

| Display | Command | Output |
|---|---|---|
| Discrimination against Reference A, with the difference from the context baseline | `python scripts/manuscript/main_table_discrimination.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --cohort-dir $ART/cohort --folds-dir $ART/folds/fold_csvs --out-dir manuscript` | `main_table_discrimination` |
| Context baseline, raw labels, and every test fold | `python scripts/manuscript/supplementary_tables_discrimination.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --cohort-dir $ART/cohort --folds-dir $ART/folds/fold_csvs --out-dir manuscript` | `supplementary_table_discrimination_baseline`, `supplementary_table_discrimination_raw_labels`, `supplementary_table_discrimination_folds` |
| Scores by Reference A status and grade | `python scripts/manuscript/main_figure_scores_by_grade.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --folds-dir $ART/folds/fold_csvs --out-dir manuscript` | `main_figure_scores_by_grade` |
| Discrimination by recorded grade | `python scripts/manuscript/supplementary_figure_discrimination_by_grade.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --folds-dir $ART/folds/fold_csvs --out-dir manuscript` | `supplementary_figure_discrimination_by_grade` |
| ROC curves of every test fold | `python scripts/manuscript/supplementary_figure_roc_curves.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --folds-dir $ART/folds/fold_csvs --out-dir manuscript` | `supplementary_figure_roc_curves` |
| Discrimination with each site held out | `python scripts/manuscript/main_figure_site_transportability.py --predictions $ART/predictions.parquet --cohort-slides $ART/cohort/cohort_slides.csv --folds-dir $ART/folds/fold_csvs --out-dir manuscript` | `main_figure_site_transportability`, `supplementary_table_site_transportability`, `supplementary_table_site_average_comparison`, `supplementary_table_site_comparison` |
| Aggregators, encoders and training-set size | `python scripts/manuscript/main_figure_modelling_choices.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --folds-dir $ART/folds/fold_csvs --runs-dir $RUNS --out-dir manuscript` | `main_figure_modelling_choices`, `supplementary_table_aggregators`, `supplementary_table_encoders`, `supplementary_table_learning_curve` |
| Learning curves of every finding | `python scripts/manuscript/supplementary_figure_learning_curves.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --folds-dir $ART/folds/fold_csvs --runs-dir $RUNS --out-dir manuscript` | `supplementary_figure_learning_curves` |
| Learning curves against the positive training slides | `python scripts/manuscript/supplementary_figure_learning_curve_cases.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --folds-dir $ART/folds/fold_csvs --runs-dir $RUNS --out-dir manuscript` | `supplementary_figure_learning_curve_cases` |
| Discrimination within age groups, scan sources, tissue amounts and post-mortem intervals | `python scripts/manuscript/main_figure_strata.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --cohort-dir $ART/cohort --folds-dir $ART/folds/fold_csvs --out-dir manuscript` | `main_figure_strata`, `supplementary_table_strata`, `supplementary_table_strata_adjusted` |
| Discrimination within sites and years of death | `python scripts/manuscript/supplementary_figure_strata.py --predictions $ART/predictions.parquet --reference-slides $ART/reference_a/reference_a_slide.parquet --cohort-dir $ART/cohort --folds-dir $ART/folds/fold_csvs --out-dir manuscript` | `supplementary_figure_strata` |

Rules as run. Every evaluation reads the test rows of the predictions only,
and a cell whose mask is 0 is left out for that finding. The headline model
is CLAM-MB on Virchow2 features, five-fold, with the corrected labels.

- Discrimination: AUROC, AP and prevalence per finding in every test fold,
  summarized as the mean and the sample SD over the folds (`--summary t`:
  the mean with a 95% t interval). The macro average of an organ gives each
  finding equal weight within a fold.
- Severity: each grade group against the report-negative slides of the same
  fold. Severe and extensive form one group; a positive slide without a grade
  forms its own group.
- Context baseline: an L2-regularized logistic regression per finding and
  fold, fitted on the training slides, on site, scan source, scanner
  magnification, death category, age in months, post-mortem interval, year
  of death and the logarithm of the Virchow2 tile count. Imputation, encoding
  and scaling are fitted on the training slides only.
- Raw labels: the same predictions against the raw labels, without
  retraining.
- Site transfer: site-held-out and five-fold predictions on their common
  eligible slides. AUROC and AP per site; the site average gives equal weight
  to each site with at least 10 positive and 10 negative slides for the
  finding; the pooled estimate over all slides is a sensitivity analysis.
- Modelling choices: each aggregator on Virchow2 features minus CLAM-MB, and
  each encoder with CLAM-MB minus Virchow2, on the pooled predictions of the
  slides that both score. The learning curve gives the mean and SD over folds
  at each training fraction, with the training counts that each run records
  in `data_counts.json`.
- Strata: AUROC and AP from the predictions pooled over folds within age
  groups, scan sources, tertiles of the Virchow2 tile count among the
  training-cohort slides of the organ, and post-mortem intervals of at most 6,
  more than 6 to 24, and more than 24 hours; sites and years of death in the
  supplement. A level needs at least 10 positive and 10 negative slides. The
  covariate-adjusted AUROC averages the AUROC within levels, weighted by the
  share of positive slides in each level; the prevalence-only AUROC scores
  each slide with the prevalence of its level.
- Intervals: 95% percentile intervals from 1,000 case-clustered bootstrap
  resamples (`--bootstrap`) with seed 20260916 (`--seed`). A resample keeps
  every slide of a drawn case. A paired comparison uses the same draws for
  both sides; a site comparison draws cases within each site.

The later parts (reader study, manuscript evidence) are added here as they
are released.
