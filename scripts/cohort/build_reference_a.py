"""Build Reference A: the report labels per case unit and per training slide.

Reads the findings table, the extraction schema, the cohort tables, the
direct-test screen and one label configuration per organ group. Writes
``reference_a_case.parquet`` (per case, organ, slide source and finding:
the status the central laboratory's description asserts, with severity,
extent and modifiers) and ``reference_a_slide.parquet`` (per training
slide, finding and variant: the label, the keep mask and the reason for
a mask).
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from champs_pipeline.data_prep.cohort import check_cohort
from champs_pipeline.data_prep.morphology_labels import load_schema
from champs_pipeline.data_prep.reference_a import case_reference, slide_reference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--findings", required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--screen", required=True, help="configs/cohort/screen_cases.csv")
    ap.add_argument("--config", type=Path, action="append", required=True,
                    help="A label configuration; repeat per organ group.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    findings = pd.read_parquet(args.findings)
    schema = load_schema(args.schema)
    cohort = check_cohort(pd.read_csv(args.cohort_dir / "cohort_slides.csv", low_memory=False))
    screen = pd.read_csv(args.screen, dtype=str)
    case_parts, slide_parts = [], []
    for path in args.config:
        config = yaml.safe_load(path.read_text())
        slides = cohort[cohort["training"] & (cohort["organ_group"] == config["organ"])]
        reference = case_reference(findings, schema, config["organs"])
        labels = slide_reference(slides, reference, findings, screen, config["labels"],
                                 config["organs"])
        labels.insert(0, "organ_group", config["organ"])
        case_parts.append(reference)
        slide_parts.append(labels)
        summary = labels.groupby(["variant", "finding"])["mask_reason"].value_counts()
        print(f"{config['organ']}: {len(slides)} slides\n{summary.to_string()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.concat(case_parts, ignore_index=True)
    slides = pd.concat(slide_parts, ignore_index=True)
    for column in [c for c in slides.columns if c.startswith("mod_")]:
        slides[column] = slides[column].astype("string")
    cases.to_parquet(args.out_dir / "reference_a_case.parquet", index=False)
    slides.to_parquet(args.out_dir / "reference_a_slide.parquet", index=False)
    print(f"reference_a_case.parquet: {len(cases)} rows; "
          f"reference_a_slide.parquet: {len(slides)} rows")


if __name__ == "__main__":
    main()
