"""Render the supplementary discrimination tables.

The tables are the context-baseline comparison, the raw-label sensitivity
analysis beside the corrected labels, and the estimates of every test fold.
"""

import argparse
from pathlib import Path

import pandas as pd

from champs_pipeline.eval.discrimination import summarise_metrics
from champs_pipeline.eval.display_data import (display_inputs, estimate, evaluation_data,
                                               summary_label)
from champs_pipeline.figures.labels import FINDINGS, ORGANS
from champs_pipeline.figures.output import write_table


def positives(row):
    """Positive slides with their share of the scored slides."""
    share = row["positive_slides"] / row["slides_scored"]
    return f"{int(row['positive_slides']):,} ({share:.1%})"


def baseline_table(model, baseline, summary="sd"):
    """Model and context-baseline metrics for every finding and the macro average."""
    model = model.set_index(["organ_group", "finding"])
    baseline = baseline.set_index(["organ_group", "finding"])
    label = summary_label(summary)
    rows = []
    for organ, findings in FINDINGS.items():
        for finding, name in [*findings.items(), ("macro", "Macro average")]:
            value = model.loc[(organ, finding)]
            comparison = baseline.loc[(organ, finding)]
            rows.append({
                "Organ": ORGANS[organ],
                "Finding": name,
                f"Model AUROC, {label}": estimate(value, "auroc", summary),
                f"Context baseline AUROC, {label}": estimate(comparison, "auroc", summary),
                f"Model AP, {label}": estimate(value, "average_precision", summary),
                f"Context baseline AP, {label}":
                    estimate(comparison, "average_precision", summary),
            })
    return pd.DataFrame(rows)


def raw_table(corrected, raw, summary="sd"):
    """The corrected and raw-label analyses side by side for every finding."""
    corrected = corrected.set_index(["organ_group", "finding"])
    raw = raw.set_index(["organ_group", "finding"])
    label = summary_label(summary)
    rows = []
    for organ, findings in FINDINGS.items():
        for finding, name in findings.items():
            corrected_value = corrected.loc[(organ, finding)]
            raw_value = raw.loc[(organ, finding)]
            rows.append({
                "Organ": ORGANS[organ],
                "Finding": name,
                "Corrected slides scored": f"{int(corrected_value['slides_scored']):,}",
                "Corrected positive slides (prevalence)": positives(corrected_value),
                "Raw slides scored": f"{int(raw_value['slides_scored']):,}",
                "Raw positive slides (prevalence)": positives(raw_value),
                f"AUROC corrected, {label}": estimate(corrected_value, "auroc", summary),
                f"AUROC raw, {label}": estimate(raw_value, "auroc", summary),
                f"Average precision corrected, {label}":
                    estimate(corrected_value, "average_precision", summary),
                f"Average precision raw, {label}":
                    estimate(raw_value, "average_precision", summary),
            })
    return pd.DataFrame(rows)


def fold_table(model_folds, baseline_folds):
    """Model and context-baseline metrics of every finding in every test fold."""
    model = model_folds.set_index(["organ_group", "finding", "fold"])
    baseline = baseline_folds.set_index(["organ_group", "finding", "fold"])
    rows = []
    for organ, findings in FINDINGS.items():
        for finding, name in findings.items():
            folds = sorted(model.loc[(organ, finding)].index)
            for fold in folds:
                value = model.loc[(organ, finding, fold)]
                comparison = baseline.loc[(organ, finding, fold)]
                rows.append({
                    "Organ": ORGANS[organ],
                    "Finding": name,
                    "Fold": str(fold + 1),
                    "Slides scored": f"{int(value['slides_scored']):,}",
                    "Positive slides": f"{int(value['positive_slides']):,}",
                    "Model AUROC": f"{value['auroc']:.3f}",
                    "Context baseline AUROC": f"{comparison['auroc']:.3f}",
                    "Model AP": f"{value['average_precision']:.3f}",
                    "Context baseline AP": f"{comparison['average_precision']:.3f}",
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--reference-slides", type=Path, required=True)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--folds-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--summary", choices=["sd", "t"], default="sd")
    ap.add_argument("--draft", action="store_true",
                    help="render from predictions without a complete inference record; "
                         "the provenance records the display as a draft")
    args = ap.parse_args()

    elig = evaluation_data(args.predictions, args.reference_slides, args.cohort_dir,
                           args.folds_dir, args.summary, draft=args.draft)
    raw = evaluation_data(args.predictions, args.reference_slides, args.cohort_dir,
                          args.folds_dir, args.summary, evaluation_variant="raw",
                          include_baseline=False, draft=args.draft)
    raw_summary, _, _ = summarise_metrics(raw["scored"], summary=args.summary)
    inputs = display_inputs(args.predictions, args.reference_slides, args.cohort_dir,
                            args.folds_dir, args.draft)
    tables = [
        ("supplementary_table_discrimination_baseline",
         baseline_table(elig["model_summary"], elig["baseline_summary"], args.summary),
         (r"@{}l@{\hskip5pt}>{\raggedright\arraybackslash}p{3.0cm}"
          r"@{\hskip5pt}r@{\hskip5pt}r@{\hskip5pt}r@{\hskip5pt}r@{}"),
         13, False),
        ("supplementary_table_discrimination_raw_labels",
         raw_table(elig["model_summary"], raw_summary, args.summary),
         (r"@{}l@{\hskip3pt}>{\raggedright\arraybackslash}p{2.75cm}"
          r"@{\hskip3pt}r@{\hskip3pt}r@{\hskip3pt}r@{\hskip3pt}r"
          r"@{\hskip3pt}r@{\hskip3pt}r@{\hskip3pt}r@{\hskip3pt}r@{}"),
         12, False),
        ("supplementary_table_discrimination_folds",
         fold_table(elig["fold_metrics"], elig["baseline_fold_metrics"]),
         "llrrrrrrr", 12, True),
    ]
    for name, result, align, header_width, long in tables:
        write_table(result, args.out_dir, name, inputs, align=align, group_column="Organ",
                    header_width=header_width, long=long, draft=args.draft)


if __name__ == "__main__":
    main()
