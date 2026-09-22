"""Render the finding supply and the candidate table from Reference A and the cohort.

``finding_supply.csv``: per study finding, the positive cases and their
share of the examined cases, the positive units, the positive units
with a linked slide and with a training slide, and the cases with at
least one positive unit that has a training slide. ``candidate_table.csv``: every finding the
central laboratory's descriptions record at least ten times in the
lungs or the liver, with its case prevalence, the share of records that
carry a severity, the cause-of-death conditions the curated map ties it
to, the share of examined cases whose causal chain holds one of those
conditions, and the selection criteria; the study findings are marked.
``cause_codes.csv``: for every row of the map, the codes of the causal
chains that fall under it.

A finding is selected when at least 250 examined cases are positive for it
and a condition that the map ties to it at the primary or the contributing
tier occurs in the causal chain of at least 1% of the examined cases with a
DeCoDe record. The selected findings must be the study findings.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from champs_pipeline.data_prep.cohort import UNIT
from champs_pipeline.data_prep.decode_results import (cases_with_codes, chain_codes,
                                                      icd10_readings, load_cause_chains,
                                                      load_icd11_correspondence,
                                                      load_icd_descriptions, matching_codes)

STUDY_FINDINGS = {
    "aspiration_squames": "lung", "aspiration_meconium": "lung", "bronchopneumonia": "lung",
    "pneumonitis": "lung", "hyaline_membranes": "lung",
    "hemozoin_pigment": "liver", "steatosis": "liver",
}
MIN_RECORDS = 10
MIN_POSITIVE_CASES = 250
MIN_CAUSE_CHAIN_SHARE = 0.01
CAUSE_TIERS = ["primary", "contributing"]


def finding_supply(reference, cohort, cases, organ_groups):
    """The sample supply per study finding."""
    positive = reference[reference["status"] == "positive"].copy()
    positive["organ_group"] = positive["organ"].map(organ_groups)
    n_examined = int(cases["examined"].sum())
    linked = set(map(tuple, cohort.loc[cohort["linked"], UNIT].values))
    training = set(map(tuple, cohort.loc[cohort["training"], UNIT].values))
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
            "cases_with_positive_training_slide": len({key[0] for key in keys if key in training}),
        })
    return pd.DataFrame(rows)


def cause_codes(decode_map, readings):
    """For each row of the cause-of-death map, the chain codes that fall under its ICD-10 field."""
    codes = {}
    rows = []
    for index, entry in decode_map.iterrows():
        codes[index] = matching_codes(entry["icd"], readings)
        rows.append({
            "finding": entry["finding"], "organ_scope": entry["organ_scope"],
            "decode_condition": entry["decode_condition"], "icd": entry["icd"],
            "tier": entry["tier"], "chain_codes": "; ".join(sorted(codes[index])),
        })
    return codes, pd.DataFrame(rows)


def candidate_table(reference, cases, decode_map, organ_groups, chains, codes_by_map_row):
    """Every finding with at least ten positive records in an organ group, with the selection."""
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
        causes = mapped[mapped["tier"].isin(CAUSE_TIERS)]
        codes = set()
        for index in causes.index:
            codes |= codes_by_map_row[index]
        chain_share = cases_with_codes(chains, codes).mean()
        n_positive = group["champs_deid"].nunique()
        rows.append({
            "finding": finding, "organ": organ_group,
            "positive_cases": n_positive,
            "case_prevalence": round(n_positive / n_examined, 4),
            "positive_units": len(group),
            "share_with_severity": round(group["severity"].notna().mean(), 3),
            "usable_extent_values": int((group["extent"].value_counts() >= MIN_RECORDS).sum()),
            "decode_tier": tier,
            "decode_conditions": "; ".join(mapped.loc[mapped["tier"] == tier, "decode_condition"]),
            "cause_conditions": "; ".join(causes["decode_condition"]),
            "cause_chain_share": round(chain_share, 4),
            "meets_case_criterion": n_positive >= MIN_POSITIVE_CASES,
            "meets_cause_criterion": chain_share >= MIN_CAUSE_CHAIN_SHARE,
            "study_finding": STUDY_FINDINGS.get(finding) == organ_group,
        })
    table = pd.DataFrame(rows).sort_values(["organ", "positive_cases"], ascending=[True, False])
    unmapped = set(table["finding"]) - set(decode_map["finding"])
    if unmapped:
        raise ValueError(f"findings missing from the cause-of-death map: {sorted(unmapped)}")
    table["selected"] = table["meets_case_criterion"] & table["meets_cause_criterion"]
    if not table["selected"].equals(table["study_finding"]):
        differing = table.loc[table["selected"] != table["study_finding"], ["finding", "organ"]]
        raise ValueError(f"the selection criteria do not select the study findings:\n{differing}")
    return table.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--decode-map", required=True, help="configs/cohort/decode_histology_map.csv")
    ap.add_argument("--decode", type=Path, required=True, help="CHAMPS_deid_decode_results.csv")
    ap.add_argument("--icd-descriptions", type=Path, required=True,
                    help="CHAMPS_icd_descriptions.csv")
    ap.add_argument("--icd11-correspondence", type=Path, required=True,
                    help="configs/cohort/icd11_to_icd10.csv")
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

    examined = cases.loc[cases["examined"], "champs_deid"].astype(str)
    chains = load_cause_chains(args.decode, examined)
    readings = icd10_readings(chain_codes(chains), load_icd_descriptions(args.icd_descriptions),
                              load_icd11_correspondence(args.icd11_correspondence))
    codes_by_map_row, codes_table = cause_codes(decode_map, readings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    supply = finding_supply(reference, cohort, cases, organ_groups)
    candidates = candidate_table(reference, cases, decode_map, organ_groups, chains,
                                 codes_by_map_row)
    supply.to_csv(args.out_dir / "finding_supply.csv", index=False)
    candidates.to_csv(args.out_dir / "candidate_table.csv", index=False)
    codes_table.to_csv(args.out_dir / "cause_codes.csv", index=False)
    print(f"examined cases with a DeCoDe record: {len(chains):,}")
    print(supply.to_string(index=False))
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
