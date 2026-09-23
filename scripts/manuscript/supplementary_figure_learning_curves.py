"""Render learning curves of AUROC and average precision for every finding."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from champs_pipeline.eval import load
from champs_pipeline.eval.discrimination import scored_slides
from champs_pipeline.eval.display_data import display_inputs
from champs_pipeline.eval.modelling_choices import (counts_path, family_predictions,
                                                    learning_curve_folds,
                                                    learning_curve_summary, training_counts)
from champs_pipeline.figures import style
from champs_pipeline.figures.labels import FINDINGS, study_findings
from champs_pipeline.figures.output import save_figure


HEIGHT_MM = 120
REFERENCE_COLUMNS = ["organ_group", "slide_id", "finding", "variant", "label", "severity"]
FRACTIONS = [0.10, 0.20, 0.35, 0.55, 0.75, 1.00]
METRIC_STYLE = {"auroc": ("AUROC", style.CATEGORICAL[0]),
                "average_precision": ("AP", style.CATEGORICAL[1])}


def panel(ax, curve, folds, organ, finding, title):
    """Draw both metrics of one finding: fold values, mean and a band of one SD."""
    rows = curve[(curve["organ_group"] == organ) & (curve["finding"] == finding)]
    rows = rows.sort_values("fraction")
    fold_rows = folds[(folds["organ_group"] == organ) & (folds["finding"] == finding)]
    x = rows["fraction"] * 100
    for metric, (label, color) in METRIC_STYLE.items():
        ax.fill_between(x, rows[f"{metric}_mean"] - rows[f"{metric}_sd"],
                        rows[f"{metric}_mean"] + rows[f"{metric}_sd"], color=color, alpha=0.18,
                        linewidth=0)
        ax.plot(fold_rows["fraction"] * 100, fold_rows[metric], marker="o", markersize=2.2,
                linestyle="none", color=color, alpha=0.45)
        ax.plot(x, rows[f"{metric}_mean"], marker="o", markersize=3.4, color=color,
                linewidth=style.LINE_WIDTH, label=label)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    ax.set_xticks([f * 100 for f in FRACTIONS], [f"{f * 100:.0f}" for f in FRACTIONS])
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", color="#E5E5E5", linewidth=style.LINE_WIDTH)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8, length=2)


def figure(curve, folds):
    """Draw one panel per finding of both organs with a shared legend."""
    order = study_findings()
    fig = plt.figure(figsize=style.figure_size(style.FULL_WIDTH_MM, HEIGHT_MM))
    grid = fig.add_gridspec(2, 4, left=0.07, right=0.99, bottom=0.17, top=0.93,
                            wspace=0.28, hspace=0.5)
    axes = []
    for index, (organ, finding) in enumerate(order):
        ax = fig.add_subplot(grid[index // 4, index % 4])
        panel(ax, curve, folds, organ, finding, FINDINGS[organ][finding])
        if index % 4:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel("AUROC or AP", fontsize=8, labelpad=2)
        if index >= 3:
            ax.set_xlabel("Training cases (%)", fontsize=8, labelpad=2)
        style.panel_label(ax, chr(ord("a") + index), dx_pt=-14)
        axes.append(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2,
               fontsize=8, frameon=False)
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
    folds = learning_curve_folds(scored)

    inputs = display_inputs(args.predictions, args.reference_slides, folds_dir=args.folds_dir,
                            draft=args.draft)
    inputs += [counts_path(args.runs_dir, run_id) for run_id in run_ids]
    style.apply()
    fig = figure(curve, folds)
    save_figure(fig, args.out_dir, "supplementary_figure_learning_curves",
                {"": curve[curve["finding"] != "macro"]}, inputs, draft=args.draft)


if __name__ == "__main__":
    main()
