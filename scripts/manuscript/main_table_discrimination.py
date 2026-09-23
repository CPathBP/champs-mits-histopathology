"""Render the main cross-validated discrimination table."""

import argparse
from pathlib import Path

import pandas as pd

from champs_pipeline.eval.display_data import (display_inputs, estimate, evaluation_data,
                                               summary_label)
from champs_pipeline.figures.labels import FINDINGS, ORGANS
from champs_pipeline.figures.output import write_table


def table(model, difference, summary="sd"):
    """One row per finding and one macro-average row per organ."""
    model = model.set_index(["organ_group", "finding"])
    difference = difference.set_index(["organ_group", "finding"])
    label = summary_label(summary)
    rows = []
    for organ, findings in FINDINGS.items():
        for finding, name in [*findings.items(), ("macro", "Macro average")]:
            value = model.loc[(organ, finding)]
            if finding == "macro":
                counts = "–"
            else:
                prevalence = value["positive_slides"] / value["slides_scored"]
                counts = (f"{int(value['positive_slides']):,}/{int(value['slides_scored']):,}"
                          f" ({prevalence:.1%})")
            rows.append({
                "Organ": ORGANS[organ],
                "Finding": name,
                "Positive/scored slides": counts,
                f"Model AUROC, {label}": estimate(value, "auroc", summary),
                f"Model AP, {label}": estimate(value, "average_precision", summary),
                f"AUROC difference vs context baseline, {label}":
                    estimate(difference.loc[(organ, finding)], "difference", summary),
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

    values = evaluation_data(args.predictions, args.reference_slides, args.cohort_dir,
                             args.folds_dir, args.summary, draft=args.draft)
    inputs = display_inputs(args.predictions, args.reference_slides, args.cohort_dir,
                            args.folds_dir, args.draft)
    write_table(table(values["model_summary"], values["difference"], args.summary),
                args.out_dir, "main_table_discrimination", inputs,
                align=(r"@{}l@{\hskip6pt}>{\raggedright\arraybackslash}p{3.2cm}"
                       r"@{\hskip6pt}r@{\hskip6pt}r@{\hskip6pt}r@{\hskip6pt}r@{}"),
                group_column="Organ", header_width=12, draft=args.draft)


if __name__ == "__main__":
    main()
