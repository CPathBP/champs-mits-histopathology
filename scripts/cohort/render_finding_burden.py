"""Render the finding supply and the candidate table from Reference A and the cohort.

``finding_supply.csv``: per study finding, the positive cases and their
share of the examined cases, the positive units, the positive units
with a linked slide and with a training slide, and the cases with at
least one training slide. ``candidate_table.csv``: every finding the
central laboratory's descriptions record at least ten times in the
lungs or the liver, with its case prevalence, the share of records that
carry a severity, and the cause-of-death conditions the curated map
ties it to; the study findings are marked.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from champs_pipeline.data_prep.cohort import UNIT

STUDY_FINDINGS = {
    "aspiration_squames": "lung", "aspiration_meconium": "lung", "bronchopneumonia": "lung",
    "pneumonitis": "lung", "hyaline_membranes": "lung",
    "hemozoin_pigment": "liver", "steatosis": "liver",
}
MIN_RECORDS = 10


def finding_supply(reference, cohort, cases, organ_groups):
    """The sample supply per study finding."""
    positive = reference[reference["status"] == "positive"].copy()
    positive["organ_group"] = positive["organ"].map(organ_groups)
    n_examined = int(cases["examined"].sum())
    linked = set(map(tuple, cohort.loc[cohort["linked"], UNIT].values))
    training = set(map(tuple, cohort.loc[cohort["training"], UNIT].values))
    training_cases = set(cohort.loc[cohort["training"], "champs_deid"])
    rows = []
    for finding, organ_group in STUDY_FINDINGS.items():
        units = positive[(positive["finding"] == finding)
                         & (positive["organ_group"] == organ_group)]
        keys = list(map(tuple, units[UNIT].values))
        n_cases = units["champs_deid"].nunique()
        rows.append({
            "finding": finding, "organ": organ_group,
            "positive_cases": n_cases, "case_prevalence": round(n_cases / n_examined, 4),
            "positive_units": len(units),
            "units_with_linked_slide": sum(k in linked for k in keys),
            "units_with_training_slide": sum(k in training for k in keys),
            "cases_with_training_slide": len(set(units["champs_deid"]) & training_cases),
        })
    return pd.DataFrame(rows)


def candidate_table(reference, cases, decode_map, organ_groups):
    """Every finding with at least ten positive records in an organ group."""
    positive = reference[reference["status"] == "positive"].copy()
    positive["organ_group"] = positive["organ"].map(organ_groups)
    n_examined = int(cases["examined"].sum())
    rows = []
    for (finding, organ_group), group in positive.groupby(["finding", "organ_group"]):
        if len(group) < MIN_RECORDS:
            continue
        mapped = decode_map[(decode_map["finding"] == finding)
                            & decode_map["organ_scope"].isin([organ_group, "any"])]
        tier = "none"
        for candidate in ("primary", "contributing"):
            if (mapped["tier"] == candidate).any():
                tier = candidate
                break
        rows.append({
            "finding": finding, "organ": organ_group,
            "positive_cases": group["champs_deid"].nunique(),
            "case_prevalence": round(group["champs_deid"].nunique() / n_examined, 4),
            "positive_units": len(group),
            "share_with_severity": round(group["severity"].notna().mean(), 3),
            "usable_extent_values": int((group["extent"].value_counts() >= MIN_RECORDS).sum()),
            "decode_tier": tier,
            "decode_conditions": "; ".join(mapped.loc[mapped["tier"] == tier, "decode_condition"]),
            "study_finding": STUDY_FINDINGS.get(finding) == organ_group,
        })
    table = pd.DataFrame(rows).sort_values(["organ", "positive_cases"], ascending=[True, False])
    unmapped = set(table["finding"]) - set(decode_map["finding"])
    if unmapped:
        raise ValueError(f"findings missing from the cause-of-death map: {sorted(unmapped)}")
    return table.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--decode-map", required=True, help="configs/cohort/decode_histology_map.csv")
    ap.add_argument("--config", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    organ_groups = {}
    for path in args.config:
        config = yaml.safe_load(path.read_text())
        organ_groups.update({organ: config["organ"] for organ in config["organs"]})
    reference = pd.read_parquet(args.reference_dir / "reference_a_case.parquet")
    cohort = pd.read_csv(args.cohort_dir / "cohort_slides.csv", low_memory=False,
                         dtype={"champs_deid": str})
    cases = pd.read_csv(args.cohort_dir / "cohort_cases.csv", low_memory=False)
    decode_map = pd.read_csv(args.decode_map)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    supply = finding_supply(reference, cohort, cases, organ_groups)
    candidates = candidate_table(reference, cases, decode_map, organ_groups)
    supply.to_csv(args.out_dir / "finding_supply.csv", index=False)
    candidates.to_csv(args.out_dir / "candidate_table.csv", index=False)
    print(supply.to_string(index=False))
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
