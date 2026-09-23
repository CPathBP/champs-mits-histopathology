"""Render the learning curve of every finding against its positive training slides.

This is the supplementary counterpart of part c of the modelling-choices
figure. That part puts the training fraction on the x axis, which is the same
for every finding and hides how much positive evidence each finding was
trained on. Here the x axis is the mean number of report-positive training
slides per fold, as recorded by each run, so each curve is placed on the
amount of evidence behind it.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from champs_pipeline.eval import load
from champs_pipeline.eval.discrimination import scored_slides
from champs_pipeline.eval.display_data import display_inputs
from champs_pipeline.eval.modelling_choices import (counts_path, family_predictions,
                                                    learning_curve_summary, training_counts)
from champs_pipeline.figures import style
from champs_pipeline.figures.labels import FINDINGS, study_findings
from champs_pipeline.figures.output import save_figure


HEIGHT_MM = 95
WIDTH_MM = 120
REFERENCE_COLUMNS = ["organ_group", "slide_id", "finding", "variant", "label", "severity"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
SHOWN_COLUMNS = ["organ_group", "finding", "fraction", "training_positives", "training_cases",
                 "training_slides", "auroc_mean", "auroc_sd"]


def panel(ax, curve):
    """Draw the AUROC of every finding against its mean positive training slides."""
    order = study_findings()
    colors = list(style.CATEGORICAL) + [style.BLACK]
    for index, (organ, finding) in enumerate(order):
        rows = curve[(curve["organ_group"] == organ) & (curve["finding"] == finding)]
        rows = rows.sort_values("training_positives")
        ax.plot(rows["training_positives"], rows["auroc_mean"],
                marker=MARKERS[index % len(MARKERS)], markersize=3.6, color=colors[index],
                linewidth=style.LINE_WIDTH, linestyle="-" if organ == "lung" else "--",
                label=FINDINGS[organ][finding])
    ax.set_xlabel("Report-positive training slides per fold (n)", fontsize=8, labelpad=2)
    ax.set_ylabel("AUROC", fontsize=8, labelpad=2)
    low = curve.loc[curve["finding"] != "macro", "auroc_mean"].min()
    ax.set_ylim(np.floor((low - 0.06) * 20) / 20, 1.0)
    ax.grid(axis="y", color="#E5E5E5", linewidth=style.LINE_WIDTH)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8, length=2)
    ax.legend(loc="lower right", fontsize=8, frameon=False, handlelength=1.8,
              handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)


def figure(curve):
    """Draw the single panel on a figure of its own size."""
    fig = plt.figure(figsize=style.figure_size(WIDTH_MM, HEIGHT_MM))
    ax = fig.add_axes([0.13, 0.14, 0.85, 0.83])
    panel(ax, curve)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--reference-slides", type=Path, required=True)
    ap.add_argument("--folds-dir", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--draft", action="store_true",
                    help="render from predictions without a complete inference record; "
                         "the provenance records the display as a draft")
    args = ap.parse_args()

    predictions = load.predictions(args.predictions).check(draft=args.draft).frame
    reference = pd.read_parquet(args.reference_slides, columns=REFERENCE_COLUMNS)
    selected = family_predictions(predictions, "learning_curve", args.folds_dir)
    scored = scored_slides(selected, reference, "elig")
    run_ids = sorted(selected["run_id"].unique())
    curve = learning_curve_summary(scored, training_counts(run_ids, args.runs_dir))

    inputs = display_inputs(args.predictions, args.reference_slides, folds_dir=args.folds_dir,
                            draft=args.draft)
    inputs += [counts_path(args.runs_dir, run_id) for run_id in run_ids]
    style.apply()
    fig = figure(curve)
    shown = curve.loc[curve["finding"] != "macro", SHOWN_COLUMNS]
    save_figure(fig, args.out_dir, "supplementary_figure_learning_curve_cases", {"": shown},
                inputs, draft=args.draft)


if __name__ == "__main__":
    main()
