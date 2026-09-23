"""Render site-held-out discrimination and its paired five-fold comparison."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from champs_pipeline.eval import load
from champs_pipeline.eval.transportability import (
    per_site_metrics,
    pooled_comparison,
    site_average_comparison,
    paired_test_predictions,
)
from champs_pipeline.figures import style
from champs_pipeline.figures.labels import FINDINGS, ORGANS, SITES, site_code, study_findings
from champs_pipeline.figures.output import save_figure, write_table


HEIGHT_MM = 108
COLUMNS = 4


def _finding_order(frame):
    """The study findings present in ``frame``, in display order."""
    present = set(zip(frame["organ_group"], frame["finding"]))
    return [pair for pair in study_findings() if pair in present]


def _axis_floor(site, comparison):
    """Shared lower AUROC limit, at most 0.7 and below every drawn interval."""
    estimable = site.loc[site["estimable"], "auroc_ci_low"]
    summaries = comparison[["site_held_out_auroc", "fivefold_auroc"]].stack()
    low = min(estimable.min(), summaries.min())
    return min(0.7, np.floor(low * 20) / 20)


def _forest(forest, block, summary, site_order):
    """Per-site AUROC with its interval, then the two site averages in the last row."""
    for position, code in enumerate(site_order):
        row = block.loc[code]
        if not bool(row.estimable):
            continue
        forest.errorbar(
            row.auroc, position,
            xerr=[[max(0.0, row.auroc - row.auroc_ci_low)],
                  [max(0.0, row.auroc_ci_high - row.auroc)]],
            fmt="o", color=style.MODEL, markersize=3.8, capsize=0,
            linewidth=style.LINE_WIDTH,
        )
    if summary["sites"] == 0:
        return
    average_position = len(site_order)
    forest.plot([summary.site_held_out_auroc, summary.fivefold_auroc],
                [average_position, average_position], color=style.GREY,
                linewidth=style.LINE_WIDTH)
    forest.plot(summary.site_held_out_auroc, average_position, marker="D",
                markersize=5, color=style.MODEL, linestyle="none")
    forest.plot(summary.fivefold_auroc, average_position, marker="o",
                markersize=4.8, markerfacecolor="white",
                markeredgecolor=style.BLACK, markeredgewidth=style.LINE_WIDTH,
                linestyle="none")


def _difference_panel(difference, comparison, order):
    """The paired site-average difference of every finding, one row per forest panel."""
    rows = comparison.set_index(["organ_group", "finding"]).loc[order].reset_index()
    positions = np.arange(len(rows))
    difference.errorbar(
        rows["difference_auroc"], positions,
        xerr=[rows["difference_auroc"] - rows["difference_auroc_ci_low"],
              rows["difference_auroc_ci_high"] - rows["difference_auroc"]],
        fmt="o", color=style.MODEL, markersize=4, capsize=0,
        linewidth=style.LINE_WIDTH,
    )
    difference.axvline(0, color=style.BLACK, linewidth=style.LINE_WIDTH)
    # The rows follow the findings of the forest panels, so they carry those panel
    # letters instead of abbreviated finding names, which do not fit the column.
    difference.set_yticks(positions, [chr(ord("a") + position) for position in positions])
    for label in difference.get_yticklabels():
        label.set_fontweight("bold")
    difference.set_ylim(len(rows) - 0.5, -0.5)
    extent = max(abs(rows["difference_auroc_ci_low"].min()),
                 abs(rows["difference_auroc_ci_high"].max()), 0.01)
    difference.set_xlim(-extent * 1.15, extent * 1.15)
    difference.set_xlabel("Five-fold minus site-held-out", fontsize=8, labelpad=2)
    difference.grid(axis="x", color="#E5E5E5", linewidth=style.LINE_WIDTH)
    difference.set_axisbelow(True)
    difference.tick_params(axis="both", labelsize=8, length=2)
    difference.set_title("Paired AUROC difference", fontsize=8, fontweight="bold", pad=3)


def _main_figure(site, comparison):
    order = _finding_order(comparison)
    site_order = [code for code in SITES if code in set(site["site"])]
    floor = _axis_floor(site, comparison)
    fig = plt.figure(figsize=style.figure_size(style.FULL_WIDTH_MM, HEIGHT_MM))
    grid = fig.add_gridspec(2, COLUMNS, left=0.12, right=0.985, bottom=0.19, top=0.91,
                            wspace=0.24, hspace=0.48)
    forests = []
    for index in range(len(order)):
        forests.append(fig.add_subplot(grid[index], sharex=forests[0] if forests else None))
    difference = fig.add_subplot(grid[len(order)])

    y_positions = np.arange(len(site_order) + 1)
    site_labels = list(site_order) + ["Site average"]
    for index, ((organ, finding), forest) in enumerate(zip(order, forests)):
        block = site[(site["organ_group"] == organ) &
                     (site["finding"] == finding)].set_index("site")
        summary = comparison[(comparison["organ_group"] == organ) &
                             (comparison["finding"] == finding)].iloc[0]
        _forest(forest, block, summary, site_order)
        forest.set_title(FINDINGS[organ][finding], fontsize=8, fontweight="bold", pad=3)
        forest.set_yticks(y_positions, site_labels)
        forest.set_ylim(len(site_order) + 0.5, -0.5)
        forest.set_xlim(floor, 1.005)
        forest.grid(axis="x", color="#E5E5E5", linewidth=style.LINE_WIDTH)
        forest.set_axisbelow(True)
        forest.tick_params(axis="both", labelsize=8, length=2)
        # Every forest keeps its tick labels: the panel above the difference panel
        # would otherwise be read against the difference scale.
        if index >= len(order) - COLUMNS + 1:
            forest.set_xlabel("AUROC", fontsize=8, labelpad=2)
        if index % COLUMNS == 0:
            forest.get_yticklabels()[-1].set_fontweight("bold")
        else:
            forest.tick_params(axis="y", left=False, labelleft=False)
    _difference_panel(difference, comparison, order)

    forests[0].plot([], [], marker="D", color=style.MODEL, linestyle="none",
                    label="Site-held-out, site average")
    forests[0].plot([], [], marker="o", markerfacecolor="white",
                    markeredgecolor=style.BLACK, markeredgewidth=style.LINE_WIDTH,
                    linestyle="none", label="Five-fold, site average")
    handles, labels = forests[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=2, fontsize=8, frameon=False)
    for index, axis in enumerate([*forests, difference]):
        style.panel_label(axis, chr(ord("a") + index), dx_pt=-14)
    return fig


def _interval(value, low, high):
    """``value (low to high)``, or not estimable when the value is missing."""
    if pd.isna(value):
        return "Not estimable"
    return f"{value:.3f} ({low:.3f} to {high:.3f})"


def _site_table(site):
    result = site.copy()
    result["Organ"] = result["organ_group"].map(ORGANS)
    result["Finding"] = [FINDINGS[o][f] for o, f in zip(result.organ_group, result.finding)]
    result["Site"] = result["site"].map(SITES)
    result["Cases"] = result["cases"].map(lambda value: f"{value:,}")
    result["Positive/scored slides"] = result.apply(
        lambda row: f"{row.positive_slides:,}/{row.slides_scored:,} ({row.prevalence:.1%})",
        axis=1)
    result["AUROC (95% CI)"] = result.apply(
        lambda row: _interval(row.auroc, row.auroc_ci_low, row.auroc_ci_high), axis=1)
    result["AP (95% CI)"] = result.apply(
        lambda row: _interval(row.average_precision, row.average_precision_ci_low,
                              row.average_precision_ci_high), axis=1)
    return result[["Organ", "Finding", "Site", "Cases", "Positive/scored slides",
                   "AUROC (95% CI)", "AP (95% CI)"]]


def _signed(value):
    """A signed difference; a rounded zero is written without a sign."""
    if abs(value) < 0.0005:
        return "0.000"
    return f"{value:+.3f}"


def _unsigned(value):
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.3f}"


def _difference(value, low, high):
    if pd.isna(value):
        return "Not estimable"
    return f"{_signed(value)} ({_unsigned(low)} to {_unsigned(high)})"


def _with_interval(row, name):
    """The estimate called ``name`` and its interval ends; missing ones are NaN."""
    return [getattr(row, f"{name}{suffix}", np.nan) for suffix in ("", "_ci_low", "_ci_high")]


def _comparison_table(comparison):
    rows = []
    for row in comparison.itertuples(index=False):
        common = {
            "Organ": ORGANS[row.organ_group],
            "Finding": FINDINGS[row.organ_group][row.finding],
            "Positive/scored slides": (
                f"{row.positive_slides:,}/{row.slides_scored:,} ({row.prevalence:.1%})"),
        }
        if hasattr(row, "sites"):
            common["Sites averaged"] = f"{row.sites} of {row.sites_total}"
        for metric, label in [("auroc", "AUROC"), ("average_precision", "AP")]:
            rows.append({
                **common,
                "Metric": label,
                "Five-fold (95% CI)": _interval(*_with_interval(row, f"fivefold_{metric}")),
                "Site-held-out (95% CI)":
                    _interval(*_with_interval(row, f"site_held_out_{metric}")),
                "Difference (95% CI)": _difference(*_with_interval(row, f"difference_{metric}")),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cohort-slides", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--draft", action="store_true",
                        help="render from predictions without a complete inference record; "
                             "the provenance records the display as a draft")
    args = parser.parse_args()

    predictions = load.predictions(args.predictions).check(draft=args.draft).frame
    cohort_slides = pd.read_csv(args.cohort_slides, low_memory=False)
    data = paired_test_predictions(predictions, cohort_slides, args.folds_dir)
    inputs = [args.predictions]
    if not args.draft:
        inputs.append(args.predictions.with_suffix(".json"))
    inputs.append(args.cohort_slides)
    inputs.extend(sorted(args.folds_dir.glob("*_elig_*/fold_*.csv")))
    site = per_site_metrics(data, n_bootstrap=args.bootstrap, seed=args.seed)
    site["site"] = site["site"].map(site_code)
    comparison = site_average_comparison(data, n_bootstrap=args.bootstrap, seed=args.seed)
    pooled = pooled_comparison(data, n_bootstrap=args.bootstrap, seed=args.seed)

    style.apply()
    figure = _main_figure(site, comparison)
    save_figure(figure, args.out_dir, "main_figure_site_transportability",
                {"a": site, "b": comparison}, inputs, draft=args.draft)
    write_table(_site_table(site), args.out_dir, "supplementary_table_site_transportability",
                inputs, group_column="Organ", long=True, draft=args.draft)
    write_table(_comparison_table(comparison), args.out_dir,
                "supplementary_table_site_average_comparison", inputs,
                group_column="Organ", draft=args.draft)
    write_table(_comparison_table(pooled), args.out_dir,
                "supplementary_table_site_comparison", inputs,
                group_column="Organ", draft=args.draft)


if __name__ == "__main__":
    main()
