# CHAMPS MITS histopathology

Code for the analyses of the CHAMPS minimally invasive tissue sampling
histopathology study: whole-slide feature extraction, report-label
extraction, multiple-instance learning, and the evaluation against the
report-derived and the blinded reference.

- `env/`: the pinned environment, external software and model weights with
  their licences, and the checksums of the primary data (`env/EXTERNALS.md`).
- `src/champs_pipeline/`: the library.
- `scripts/`: the steps of the pipeline, one script each, grouped by part.
- `configs/`: the configuration files of the steps that take one.
- `docs/pipeline.md`: the order of operations with the command for each step.
- `tests/`: unit tests of the library and the scripts.

## Setup

```
make env          # conda environment at ./.env from env/environment.yml and env/requirements.lock
make env-check    # verify versions; with DATA_ROOT set, also the primary-data checksums
make env-gpu      # CUDA extensions for the MamMIL aggregator (needs nvcc)
make env-ssl      # packages for training the in-domain encoder
```

The primary data are not distributed here; see the data section of
`env/EXTERNALS.md`. Model weights are downloaded from their providers.
