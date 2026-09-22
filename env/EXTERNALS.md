# Software, model weights, and data

This file records the external components the analyses depend on, at the
versions used for every reported result, with their licences. Paths are
given relative to the data directory (`DATA_ROOT`) and the model cache
(`HF_HOME`) configured for the installation.

## Python environment

`make env` builds the environment and `make env-check` verifies it.
`environment.yml` provides Python 3.10.19 and the native libraries
(OpenSlide 3.4.1, libtiff 4.5.1, OpenJPEG 2.5.0). `requirements.lock` pins
every Python package at the version used; it is installed without
dependency resolution. `pip check` then reports the declared requirements
that are deliberately not met: `opencv-python-headless` 4.13.0.92 declares
`numpy>=2` alongside numpy 1.26.4 (functional, because extensions built
against the NumPy 2 headers run under NumPy 1.x), and Trident and the
DINOv2 fork declare the dependency sets of their own upstream releases,
which the lock supersedes. `make env-check` accepts exactly these lines.
`requirements-gpu-build.txt` holds `mamba-ssm`, which compiles CUDA
extensions at install time and is needed for the MamMIL aggregator only
(`make env-gpu`). `requirements-ssl.txt` holds the packages DINOv2's training
harness needs for the in-domain encoder (`make env-ssl`).

Core versions: torch 2.10.0 (CUDA 12.8), PyTorch Lightning 2.5.6,
timm 1.0.27, pylance 4.0.1, statsmodels 0.15.0.

## Whole-slide processing

Feature extraction uses Trident v0.2.3 (commit `adf3b7e8`,
CC BY-NC-ND 4.0), installed unmodified from the upstream repository at that
tag. The package does not ship its batch script, so `scripts/slides/extract_features.py`
drives Trident's `Processor` with the settings below. Two adaptations are
applied at runtime (`champs_pipeline.encoders.trident_adaptations`) rather
than by changing Trident's source, which its licence does not permit to
redistribute in modified form:

- the H-optimus-0 and H-optimus-1 loaders check for `timm == 0.9.16`; the
  check is satisfied while they build their model, and both encoders load and
  produce the stored features under timm 1.0.27 (verified on the stored
  features of the study);
- the in-domain encoder (below) is added to Trident's encoder registry so
  that it runs through the same extraction path as the public encoders.

Settings: segmentation with the `hest` segmenter at confidence 0.5, holes
treated as tissue; tiles at 20x with the tile size per encoder as stated in
Methods; no overlap. Tissue segmentation uses the DeepLabv3
checkpoint `deeplabv3_seg_v4.ckpt` from `MahmoodLab/hest-tissue-seg`
(commit `4d24a57a`, CC BY-NC-SA 4.0, SHA-256 prefix `4ddf8be82384544a`),
at 10x, with holes treated as tissue. Scanner magnification and resolution
are read from each file; the extraction manifests carry no resolution
column.

## In-domain encoder

The in-domain encoder was trained with DINOv2 (upstream commit `7764ea0f`,
2026-06-03, Apache 2.0), modified and published at
`github.com/CPathBP/dinov2`, tag `champs-blockexp-v1` (commit `e70a7228`),
which `make env` installs.
The modifications are the adaptation method: the warm-started H-optimus-0
backbone is extended by four identity-initialised transformer blocks, the
original blocks are frozen, and the new blocks, the final normalisation
layer, and the projection heads are trained; the KoLeo regulariser is
replaced by a kernel-density uniformity loss on the unit hypersphere
(weight 0.05); HED colour-space stain augmentation is added; and a tile
dataset reads the CHAMPS shards. Configuration:
`configs/ssl/champs_vitg14_blockexp.yaml`; eight H100 GPUs, batch
size 32 per GPU, teacher checkpoint of iteration 99,999. The fork's
resharding routine was a no-op, so the teacher network used in the
forward pass was refreshed only when its weights were gathered for
evaluation, every 12,500 iterations; the student therefore distilled a
teacher that changed eight times over the run.

| Checkpoint | SHA-256 prefix |
|---|---|
| Warm start (H-optimus-0 converted to DINOv2 layout) | `6f5c650232` |
| Trained teacher, iteration 99,999 | `adc7fd5312` |

The trained encoder derives from H-optimus-0 (Apache 2.0) and is released
under the same licence.

## Model weights

| Model | Repository | Commit | Licence | Access |
|---|---|---|---|---|
| Virchow2 | `paige-ai/Virchow2` | `31586458` | CC BY-NC-ND 4.0 | gated |
| UNI2-h | `MahmoodLab/UNI2-h` | `d517a8dd` | CC BY-NC-ND 4.0 | gated |
| CONCH v1.5 | `MahmoodLab/conchv1_5` | `3e5766a5` | CC BY-NC-ND 4.0 | gated |
| H-optimus-0 | `bioptimus/H-optimus-0` | `b145cc1e` | Apache 2.0 | gated |
| H-optimus-1 | `bioptimus/H-optimus-1` | `3592cb22` | CC BY-NC-ND 4.0 | gated, manual approval |
| Gemma 4 31B IT, NVFP4 | `nvidia/Gemma-4-31B-IT-NVFP4` | `4135a98a` | Apache 2.0 (Gemma 4 terms) | open |

Weights are downloaded from the providers; none are redistributed here.
Access to the gated models is requested from each provider and requires
agreement to the respective licence. All use in this work is
non-commercial academic research.

## Report extraction

Reports were processed with vLLM from the `vllm/vllm-openai` container,
serving the Gemma weights above with an fp8 KV cache, a context of 40,960
tokens, an output cap of 8,192 tokens, temperature 0, and the reply
constrained to the JSON schema derived from the extraction schema. No
released tag of that container serves this model, so the run used the
`nightly` image of 2026-09-12 (vLLM build `0.1.1.dev50+geed1f3d0c`),
snapshot to a local file whose checksum the run record carries; the image
itself cannot be redistributed. Quantised inference is not bit-reproducible
across hardware or engine builds; the extracted records are therefore
provided as data, and re-running the extraction reproduces them
statistically, not exactly. The run record (`run_meta.json`) stores the
engine build, the context length, the output cap, the decoding constraint,
and a fingerprint of the rendered prompt; a repeat of one shard under the
same settings measures the agreement.

## Data

### Restricted primary data

The primary data are not distributed with this repository. They are the
CHAMPS Level 2 release 4.30 (2025-09-01), the pathology report export, and
the whole-slide image archive. CHAMPS files are subject to the CHAMPS
data-use terms. The files below
are the exact inputs used; `checksums.sha256` holds their full SHA-256
digests so that a holder of the same release can verify identity with
`sha256sum -c checksums.sha256` from `DATA_ROOT`.

| File | Rows | SHA-256 prefix |
|---|---|---|
| `reports/deid_CPL_DB_Diagnosis_10.1.25_final_L2_20260306_181744.csv` | 273,756 | `ac18edd83dfe8549` |
| `reports/deid_CPL_DB_IHC_final_L2_20260306_182608.csv` | 358,742 | `9e083702f4ac89b4` |
| `reports/deid_CPL_DB_PCR_final_L2_20260306_184329.csv` | 65,527 | `2faabadc40fc8139` |
| `reports/deid_CPL_DB_SpecialStains_final_L2_20260306_190056.csv` | 88,419 | `71acef0b04c61601` |
| `level-2-deidentified-data/CHAMPS_de_identified_data/CHAMPS_deid_basic_demographics.csv` | 8,619 | `2fe73abfef40f2ea` |
| `level-2-deidentified-data/CHAMPS_de_identified_data/CHAMPS_deid_decode_results.csv` | 8,619 | `b5e80e404eb6b5dc` |
| `level-2-deidentified-data/CHAMPS_de_identified_data/CHAMPS_deid_tac_results.csv` | 8,491 | `9eaa24d4eeadc7f9` |
| `level-2-deidentified-data/CHAMPS_de_identified_data/CHAMPS_deid_lab_results.csv` | 9,229 | `ab27e7b06ae6a0c7` |
| `level-2-deidentified-data/CHAMPS_de_identified_data/CHAMPS_icd_descriptions.csv` | 20,053 | `8eb3667faf359dd1` |
| `champs_deid_studyid_mapping.csv` | 8,980 | `109542a933764833` |

Whole-slide images are held in the CHAMPS archive under `DATA_ROOT/<site>/`.

### Derived data

Derived data released with this repository (prediction tables, cohort and
fold tables, extracted report records) are subject to the same CHAMPS
data-use terms.
