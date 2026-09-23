"""Render the modelling-choices figure and its supplementary tables.

Panels a and b show the cross-validated AUROC of every aggregator and encoder
for each lung finding and the macro average, on one shared scale. Panel c
shows the learning curve of the reference configuration (CLAM-MB on Virchow2)
for every finding of both organs. The supplementary tables give absolute
metrics for every variant and finding, paired differences from the reference
configuration for both metrics, and the learning curve by organ and fraction.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from champs_pipeline.eval import load
from champs_pipeline.eval.discrimination import scored_slides
from champs_pipeline.eval.display_data import display_inputs
from champs_pipeline.eval.modelling_choices import (REFERENCES, counts_path,
                                                    family_predictions,
                                                    learning_curve_summary,
                                                    paired_differences, training_counts,
                                                    variant_summaries)
from champs_pipeline.figures import style
from champs_pipeline.figures.labels import (AGGREGATORS, ENCODERS, FINDINGS, ORGANS,
                                           study_findings)
from champs_pipeline.figures.output import save_figure, write_table


HEIGHT_MM = 90
REFERENCE_COLUMNS = ["organ_group", "slide_id", "finding", "variant", "label", "severity"]
AGGREGATOR_ORDER = ["clam_mb", "meanpool", "abmil", "acmil", "transmil", "mammil"]
ENCODER_ORDER = ["virchow2", "uni_v2", "conch_v15", "hoptimus0", "hoptimus1", "champs_mits"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
FRACTION_PERCENT = {0.10: "10", 0.20: "20", 0.35: "35", 0.55: "55", 0.75: "75", 1.00: "100"}
ROTATED_FINDINGS = {"aspiration_squames": "Squames", "aspiration_meconium": "Meconium",
                    "bronchopneumonia": "Bronchopneumonia", "pneumonitis": "Pneumonitis",
                    "hyaline_membranes": "Hyaline membranes", "macro": "Macro average"}


def compared_family(predictions, reference, folds_dir, family, n_bootstrap, seed):
    """Score one comparison family and compare every variant with the reference."""
    selected = family_predictions(predictions, family, folds_dir)
    scored = scored_slides(selected, reference, "elig")
    summary = variant_summaries(scored)
    differences = paired_differences(scored, REFERENCES[family], n_bootstrap, seed)
    return summary, differences


def learning_curve(predictions, reference, folds_dir, runs_dir):
    """Summarise the learning curve; also return the training counts files it read."""
    selected = family_predictions(predictions, "learning_curve", folds_dir)
    scored = scored_slides(selected, reference, "elig")
    run_ids = sorted(selected["run_id"].unique())
    curve = learning_curve_summary(scored, training_counts(run_ids, runs_dir))
    return curve, [counts_path(runs_dir, run_id) for run_id in run_ids]


def absolute_panel(ax, summary, order, names, organ="lung"):
    """Draw the AUROC of every variant, dodged within each finding group."""
    findings = list(FINDINGS[organ]) + ["macro"]
    variants = [variant for variant in order if variant in set(summary["variant"])]
    width = 0.72
    offsets = np.linspace(-width / 2, width / 2, len(variants)) if len(variants) > 1 else [0.0]
    table = summary.set_index(["variant", "organ_group", "finding"])
    for index, variant in enumerate(variants):
        color = style.CATEGORICAL[index % len(style.CATEGORICAL)]
        marker = MARKERS[index % len(MARKERS)]
        for position, finding in enumerate(findings):
            row = table.loc[(variant, organ, finding)]
            x = position + offsets[index]
            ax.errorbar(x, row["auroc_mean"], yerr=row["auroc_sd"], fmt=marker, color=color,
                        markersize=3.6, capsize=0, linewidth=style.LINE_WIDTH,
                        label=names[variant] if position == 0 else None)
    ax.set_xticks(range(len(findings)), [ROTATED_FINDINGS[f] for f in findings], rotation=40,
                  ha="right", rotation_mode="anchor")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(-0.6, len(findings) - 0.4)
    for boundary in range(len(findings) - 1):
        ax.axvline(boundary + 0.5, color="#E5E5E5", linewidth=style.LINE_WIDTH, zorder=0)
    ax.set_ylabel("AUROC", fontsize=8, labelpad=2)
    ax.grid(axis="y", color="#E5E5E5", linewidth=style.LINE_WIDTH)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8, length=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2, fontsize=8,
              frameon=False, handletextpad=0.3, columnspacing=1.0, borderaxespad=0)


def learning_curve_panel(ax, curve):
    """Draw the AUROC of every finding against the training fraction."""
    order = study_findings()
    colors = list(style.CATEGORICAL) + [style.BLACK]
    for index, (organ, finding) in enumerate(order):
        rows = curve[(curve["organ_group"] == organ) & (curve["finding"] == finding)]
        rows = rows.sort_values("fraction")
        ax.plot(rows["fraction"] * 100, rows["auroc_mean"], marker=MARKERS[index % len(MARKERS)],
                markersize=3.6, color=colors[index], linewidth=style.LINE_WIDTH,
                linestyle="-" if organ == "lung" else "--",
                label=FINDINGS[organ][finding])
    ax.set_xticks([f * 100 for f in FRACTION_PERCENT], list(FRACTION_PERCENT.values()))
    ax.set_xlabel("Training cases (%)", fontsize=8, labelpad=2)
    ax.set_ylabel("AUROC", fontsize=8, labelpad=2)
    # The lower limit leaves room for the legend below the lowest curve.
    low = curve.loc[curve["finding"] != "macro", "auroc_mean"].min()
    ax.set_ylim(np.floor((low - 0.06) * 20) / 20, 1.0)
    ax.grid(axis="y", color="#E5E5E5", linewidth=style.LINE_WIDTH)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8, length=2)
    ax.legend(loc="lower right", fontsize=8, frameon=False, handlelength=1.8,
              handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)


def main_figure(aggregator_summary, encoder_summary, curve):
    """Draw the aggregator, encoder and learning-curve panels side by side."""
    fig = plt.figure(figsize=style.figure_size(style.FULL_WIDTH_MM, HEIGHT_MM))
    grid = fig.add_gridspec(1, 3, left=0.075, right=0.99, bottom=0.37, top=0.93,
                            wspace=0.26, width_ratios=[1, 1, 1])
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    absolute_panel(axes[0], aggregator_summary, AGGREGATOR_ORDER, AGGREGATORS)
    axes[0].set_title("Aggregators on Virchow2 features", fontsize=8, fontweight="bold", pad=3)
    absolute_panel(axes[1], encoder_summary, ENCODER_ORDER, ENCODERS)
    axes[1].set_title("Encoders with CLAM-MB", fontsize=8, fontweight="bold", pad=3)
    summaries = (aggregator_summary, encoder_summary)
    low = min(min(s["auroc_mean"] - s["auroc_sd"]) for s in summaries)
    for axis in axes:
        axis.set_ylim(np.floor((low - 0.01) * 20) / 20, 1.0)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].set_ylabel("")
    curve_axis = fig.add_subplot(grid[2])
    learning_curve_panel(curve_axis, curve)
    curve_axis.set_title("Training-set size", fontsize=8, fontweight="bold", pad=3)
    for index, axis in enumerate([*axes, curve_axis]):
        style.panel_label(axis, chr(ord("a") + index), dx_pt=-24 if index != 1 else -10)
    return fig


def estimate(row, metric):
    """Format a mean over folds and its SD."""
    return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_sd']:.3f}"


def signed_interval(row, metric):
    """Format a paired difference with its sign and its 95% interval."""
    value = row[f"difference_{metric}"]
    # Avoid a visually misleading negative zero after rounding.
    value = 0.0 if abs(value) < 0.0005 else value
    low = row[f"difference_{metric}_ci_low"]
    high = row[f"difference_{metric}_ci_high"]
    return f"{value:+.3f} ({low:.3f} to {high:.3f})"


def variant_table(summary, differences, order, names, column, reference_variant, organ="lung"):
    """One row per variant and finding: absolute metrics and paired differences."""
    present = set(summary["variant"])
    variants = [variant for variant in order if variant in present]
    summary = summary.set_index(["variant", "organ_group", "finding"])
    differences = differences.set_index(["variant", "finding"])
    reference_name = names[reference_variant]
    auroc_column = f"AUROC difference from {reference_name} (95% CI)"
    ap_column = f"AP difference from {reference_name} (95% CI)"
    rows = []
    for variant in variants:
        for finding, label in [*FINDINGS[organ].items(), ("macro", "Macro average")]:
            value = summary.loc[(variant, organ, finding)]
            row = {column: names[variant], "Finding": label,
                   "AUROC, mean ± SD": estimate(value, "auroc"),
                   "AP, mean ± SD": estimate(value, "average_precision")}
            if variant == reference_variant:
                row[auroc_column] = "reference"
                row[ap_column] = "reference"
            else:
                comparison = differences.loc[(variant, finding)]
                row[auroc_column] = signed_interval(comparison, "auroc")
                row[ap_column] = signed_interval(comparison, "average_precision")
            rows.append(row)
    return pd.DataFrame(rows)


def learning_curve_table(curve):
    """One row per organ and training fraction with the macro-average metrics."""
    rows = []
    for organ in ORGANS:
        macro = curve[(curve["organ_group"] == organ) & (curve["finding"] == "macro")]
        for row in macro.sort_values("fraction").itertuples(index=False):
            values = row._asdict()
            rows.append({"Organ": ORGANS[organ],
                         "Training cases (%)": FRACTION_PERCENT[round(row.fraction, 2)],
                         "Training cases per fold, mean": f"{row.training_cases:,.0f}",
                         "Macro AUROC, mean ± SD": estimate(values, "auroc"),
                         "Macro AP, mean ± SD": estimate(values, "average_precision")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--reference-slides", type=Path, required=True)
    ap.add_argument("--folds-dir", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--draft", action="store_true",
                    help="render from predictions without a complete inference record; "
                         "the provenance records the display as a draft")
    args = ap.parse_args()

    predictions = load.predictions(args.predictions).check(draft=args.draft).frame
    reference = pd.read_parquet(args.reference_slides, columns=REFERENCE_COLUMNS)
    aggregator_summary, aggregator_differences = compared_family(
        predictions, reference, args.folds_dir, "aggregator_comparison", args.bootstrap,
        args.seed)
    encoder_summary, encoder_differences = compared_family(
        predictions, reference, args.folds_dir, "encoder_comparison", args.bootstrap, args.seed)
    curve, counts_files = learning_curve(predictions, reference, args.folds_dir, args.runs_dir)

    inputs = display_inputs(args.predictions, args.reference_slides, folds_dir=args.folds_dir,
                            draft=args.draft)
    curve_inputs = inputs + counts_files
    style.apply()
    figure = main_figure(aggregator_summary, encoder_summary, curve)
    save_figure(figure, args.out_dir, "main_figure_modelling_choices",
                {"a": aggregator_summary, "b": encoder_summary,
                 "c": curve[curve["finding"] != "macro"]}, curve_inputs, draft=args.draft)
    write_table(variant_table(aggregator_summary, aggregator_differences, AGGREGATOR_ORDER,
                              AGGREGATORS, "Aggregator", REFERENCES["aggregator_comparison"]),
                args.out_dir, "supplementary_table_aggregators", inputs,
                group_column="Aggregator", long=True, draft=args.draft)
    write_table(variant_table(encoder_summary, encoder_differences, ENCODER_ORDER, ENCODERS,
                              "Encoder", REFERENCES["encoder_comparison"]),
                args.out_dir, "supplementary_table_encoders", inputs,
                group_column="Encoder", long=True, draft=args.draft)
    write_table(learning_curve_table(curve), args.out_dir, "supplementary_table_learning_curve",
                curve_inputs, group_column="Organ", draft=args.draft)
    figure_dir = args.out_dir / "main_figure_modelling_choices"
    aggregator_differences.to_csv(figure_dir / "paired_differences_aggregators.csv", index=False)
    encoder_differences.to_csv(figure_dir / "paired_differences_encoders.csv", index=False)


if __name__ == "__main__":
    main()
