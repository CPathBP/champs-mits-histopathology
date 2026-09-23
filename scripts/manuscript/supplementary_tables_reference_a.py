"""Render the supplementary tables of Reference A for the study findings.

Reads ``reference_a_slide.parquet`` and writes two tables, each as LaTeX and
CSV with a provenance record. ``supplementary_table_reference_a_labels`` gives, per
finding, the positive slides, the slides masked for each reason, and the
prevalence with the raw and with the corrected labels. Prevalence is the
share of positive slides among the slides that are not masked.
``supplementary_table_reference_a_grades`` gives the grade recorded for the positive
slides and the share of positive slides with a recorded extent. All counts
are slides of the study cohort.
"""

import argparse
from pathlib import Path

import pandas as pd

from champs_pipeline.figures.labels import FINDINGS, ORGANS
from champs_pipeline.figures.output import write_table

RAW = "raw"
CORRECTED = "elig"
MASK_REASONS = {
    "uncertain": "Masked, uncertain wording",
    "eligibility": "Masked, case eligibility",
    "related_finding": "Masked, related finding",
    "quality": "Masked, quality",
}
SEVERITIES = {
    "minimal": "Minimal",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
    "extensive": "Extensive",
}


def count_share(count, total):
    return f"{int(count):,} ({count / total:.1%})"


def finding_slides(slides, organ, finding, variant):
    selected = ((slides["organ_group"] == organ) & (slides["finding"] == finding)
                & (slides["variant"] == variant))
    return slides[selected]


def label_row(organ, finding, raw, corrected):
    """Positive slides, masked slides by reason and prevalence for one finding."""
    positive = int((corrected["label"] == 1).sum())
    if positive != int((raw["label"] == 1).sum()):
        raise ValueError(f"{finding}: the label variants differ in positive slides")
    raw_reasons = set(raw.loc[raw["label"].isna(), "mask_reason"])
    if raw_reasons - {"uncertain"}:
        raise ValueError(f"{finding}: raw labels are masked for {raw_reasons}")
    masked = corrected.loc[corrected["label"].isna(), "mask_reason"].value_counts()
    unlisted = set(masked.index) - set(MASK_REASONS)
    if unlisted:
        raise ValueError(f"{finding}: mask reasons without a column: {unlisted}")

    row = {"Organ": ORGANS[organ], "Finding": FINDINGS[organ][finding],
           "Positive slides": f"{positive:,}"}
    for reason, header in MASK_REASONS.items():
        row[header] = f"{int(masked.get(reason, 0)):,}"
    row["Prevalence, raw labels"] = f"{positive / raw['label'].notna().sum():.1%}"
    row["Prevalence, corrected labels"] = f"{positive / corrected['label'].notna().sum():.1%}"
    return row


def grade_row(organ, finding, corrected):
    """The recorded grade and extent of the positive slides of one finding."""
    positives = corrected[corrected["label"] == 1]
    total = len(positives)
    unlisted = set(positives["severity"].dropna()) - set(SEVERITIES)
    if unlisted:
        raise ValueError(f"{finding}: grades without a column: {unlisted}")

    row = {"Organ": ORGANS[organ], "Finding": FINDINGS[organ][finding],
           "Positive slides": f"{total:,}"}
    for severity, header in SEVERITIES.items():
        row[header] = count_share((positives["severity"] == severity).sum(), total)
    row["Not graded"] = count_share(positives["severity"].isna().sum(), total)
    row["Extent recorded"] = count_share(positives["extent"].notna().sum(), total)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-slides", type=Path, required=True,
                    help="reference_a/reference_a_slide.parquet")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    columns = ["organ_group", "finding", "variant", "label", "mask_reason", "severity", "extent"]
    slides = pd.read_parquet(args.reference_slides, columns=columns)
    label_rows = []
    grade_rows = []
    for organ, findings in FINDINGS.items():
        for finding in findings:
            raw = finding_slides(slides, organ, finding, RAW)
            corrected = finding_slides(slides, organ, finding, CORRECTED)
            label_rows.append(label_row(organ, finding, raw, corrected))
            grade_rows.append(grade_row(organ, finding, corrected))

    labels = pd.DataFrame(label_rows)
    grades = pd.DataFrame(grade_rows)
    write_table(labels, args.out_dir, "supplementary_table_reference_a_labels",
                [args.reference_slides], align="ll" + "r" * (labels.shape[1] - 2),
                group_column="Organ", header_width=12)
    write_table(grades, args.out_dir, "supplementary_table_reference_a_grades",
                [args.reference_slides], align="ll" + "r" * (grades.shape[1] - 2),
                group_column="Organ", header_width=12)


if __name__ == "__main__":
    main()
