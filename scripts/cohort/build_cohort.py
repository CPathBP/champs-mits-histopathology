"""Build the slide cohort, the case table, the drop log and the funnel.

Reads the slide inventory, the feature index and the stain scores of the
slide arm, the findings table and the case attribution of the report
arm, the study-id mapping and the release's
demographics and cause-of-death tables. Writes ``cohort_slides.csv``
(one row per H&E slide of a target organ with its inclusion flags and
per-encoder feature paths), ``cohort_cases.csv`` (one row per case of the
report table), ``dropped_slides.csv`` (every drop with its stage and
reason) and ``funnel.json`` (the box counts of both arms).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from champs_pipeline.data_prep.cohort import (
    build_cohort, check_cohort, content_records, report_boxes, slide_table,
)
from champs_pipeline.data_prep.decode_results import load_decode_results
from champs_pipeline.data_prep.demographics import load_demographics
from champs_pipeline.data_prep.feature_index import load_feature_index


def case_table(attribution, findings, cohort, demographics, decode):
    """One row per case of the report table with its inclusion flags and covariates."""
    cases = attribution.copy()
    cases["champs_deid"] = cases["champs_deid"].astype(str)
    cases["has_records"] = cases["champs_deid"].isin(set(findings["champs_deid"]))
    cases["examined"] = cases["champs_deid"].isin(set(content_records(findings)["champs_deid"]))
    for flag in ("linked", "training"):
        counts = cohort.loc[cohort[flag]].groupby("champs_deid").size()
        cases[f"n_slides_{flag}"] = cases["champs_deid"].map(counts).fillna(0).astype(int)
        cases[flag] = cases[f"n_slides_{flag}"] > 0
    cases = cases.merge(demographics, on="champs_deid", how="left", indicator="in_demographics")
    cases["in_demographics"] = cases["in_demographics"] == "both"
    cases = cases.merge(decode, on="champs_deid", how="left")
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True, help="wsi_inventory.csv of the slide arm.")
    ap.add_argument("--index", required=True, help="feature_index.csv of the slide arm.")
    ap.add_argument("--stain-scores", required=True)
    ap.add_argument("--study-id-mapping", required=True,
                    help="CSV with study_id and champs_deid.")
    ap.add_argument("--findings", required=True, help="findings.parquet of the report arm.")
    ap.add_argument("--attribution", required=True, help="case_attribution.parquet.")
    ap.add_argument("--encoders", required=True, help="configs/cohort/encoders.yaml")
    ap.add_argument("--demographics", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    encoders = yaml.safe_load(Path(args.encoders).read_text())
    inventory = pd.read_csv(args.inventory, low_memory=False)
    mapping = pd.read_csv(args.study_id_mapping, dtype=str)
    index = load_feature_index(args.index)
    stain_scores = pd.read_csv(args.stain_scores, usecols=["slide_id", "p_non_HE"],
                               dtype={"slide_id": str})
    findings = pd.read_parquet(args.findings)
    attribution = pd.read_parquet(args.attribution)

    slides = slide_table(inventory, mapping)
    cohort, dropped, boxes = build_cohort(slides, index, stain_scores, findings,
                                          encoders["encoders"], encoders["cohort_encoder"])
    check_cohort(cohort)
    cases = case_table(attribution, findings, cohort, load_demographics(args.demographics),
                       load_decode_results(args.decode))
    funnel = {"report_arm": report_boxes(findings, attribution), "slide_arm": boxes}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.out_dir / "cohort_slides.csv", index=False)
    cases.to_csv(args.out_dir / "cohort_cases.csv", index=False)
    dropped.sort_values(["stage", "slide_id"]).to_csv(args.out_dir / "dropped_slides.csv",
                                                      index=False)
    (args.out_dir / "funnel.json").write_text(json.dumps(funnel, indent=2))
    print(pd.DataFrame(boxes).to_string(index=False))
    print(f"cohort_slides.csv: {len(cohort)} slides, {int(cohort['linked'].sum())} linked, "
          f"{int(cohort['training'].sum())} training")
    print(f"cohort_cases.csv: {len(cases)} cases, {int(cases['examined'].sum())} examined, "
          f"{int(cases['linked'].sum())} linked, {int(cases['training'].sum())} training")


if __name__ == "__main__":
    main()
