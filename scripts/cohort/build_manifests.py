"""Write the training manifests: one per organ group and label variant.

Joins the training slides of the cohort to the slide reference, the
findings table (for the negation and quality flags), the case table
(reporting template and demographics) and, when a feature store exists,
its row index. Writes ``manifest_<organ>_<variant>.csv``.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from champs_pipeline.data_prep.cohort import check_cohort
from champs_pipeline.data_prep.manifest_builder import build_manifest
from champs_pipeline.data_prep.morphology_labels import load_schema
from champs_pipeline.data_prep.reference_a import VARIANTS


def lance_indexes(lance_dir, encoders):
    """The row index of every encoder whose store exists under ``lance_dir``."""
    indexes = {}
    if lance_dir is None:
        return indexes
    for encoder in encoders:
        path = lance_dir / f"lance_index_{encoder}.csv"
        if path.exists():
            indexes[encoder] = pd.read_csv(path, dtype={"slide_id": str})
    return indexes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--findings", required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--config", type=Path, action="append", required=True)
    ap.add_argument("--encoders", required=True, help="configs/cohort/encoders.yaml")
    ap.add_argument("--lance-dir", type=Path, default=None,
                    help="Directory of the feature stores and their row indexes.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    encoders = yaml.safe_load(Path(args.encoders).read_text())
    cohort = check_cohort(pd.read_csv(args.cohort_dir / "cohort_slides.csv", low_memory=False))
    cases = pd.read_csv(args.cohort_dir / "cohort_cases.csv", low_memory=False,
                        dtype={"champs_deid": str})
    reference = pd.read_parquet(args.reference_dir / "reference_a_slide.parquet")
    findings = pd.read_parquet(args.findings)
    schema = load_schema(args.schema)
    indexes = lance_indexes(args.lance_dir, encoders["encoders"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.config:
        config = yaml.safe_load(path.read_text())
        slides = cohort[cohort["training"] & (cohort["organ_group"] == config["organ"])]
        for variant in VARIANTS:
            rows = reference[(reference["organ_group"] == config["organ"])
                             & (reference["variant"] == variant)]
            manifest = build_manifest(slides, rows, cases, findings, schema, config,
                                      encoders["encoders"], encoders["cohort_encoder"], indexes)
            out = args.out_dir / f"manifest_{config['organ']}_{variant}.csv"
            manifest.to_csv(out, index=False)
            positives = {f: int((manifest[f"label_{f}"] == 1).sum()) for f in config["labels"]}
            print(f"{out.name}: {len(manifest)} slides, {manifest['case_id'].nunique()} cases, "
                  f"positives {positives}")


if __name__ == "__main__":
    main()
