"""Render AUROC by Reference A severity group for every study finding."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from champs_pipeline.eval.discrimination import GRADE_LABELS, GRADE_ORDER, per_grade_metrics
from champs_pipeline.eval.display_data import display_inputs, evaluation_data
from champs_pipeline.figures import style
from champs_pipeline.figures.labels import FINDINGS
from champs_pipeline.figures.output import save_figure


# The layout of the scores-by-grade figure: a full-width, two-row grid with a
# small count strip beside every data panel, 100 mm high.
HEIGHT_MM = 100
POSITIVE_GRADES = GRADE_ORDER[1:]
GRADE_COLORS = dict(zip(POSITIVE_GRADES[:-1], style.ordered_colors(4, "Oranges")))
AUROC_LIMITS = (0.5, 1.0)


def source_panel(summary, fold, overall_folds, organ, finding):
    """Return plotted summaries plus every contributing fold for source data."""
    panel = summary[(summary["organ_group"] == organ) & (summary["finding"] == finding)].copy()
    panel = panel.set_index("grade_group").reindex(POSITIVE_GRADES).reset_index()
    panel = panel.rename(columns={"mean": "auroc_mean", "sd": "auroc_sd",
                                  "ci_low": "auroc_ci_low", "ci_high": "auroc_ci_high"})
    fold_panel = fold[(fold["organ_group"] == organ) & (fold["finding"] == finding)]
    for value, prefix in [("auroc", "fold_auroc"), ("grade_slides", "fold_positive_slides"),
                          ("negative_slides", "fold_report_negative_slides")]:
        wide = fold_panel.pivot(index="grade_group", columns="fold", values=value)
        wide = wide.reindex(POSITIVE_GRADES)
        wide.columns = [f"{prefix}_{int(column) + 1}" for column in wide.columns]
        panel = panel.merge(wide.reset_index(), on="grade_group", how="left")
    overall = overall_folds[(overall_folds["organ_group"] == organ)
                            & (overall_folds["finding"] == finding)].sort_values("fold")
    panel["overall_auroc_mean"] = overall["auroc"].mean()
    for row in overall.itertuples(index=False):
        panel[f"overall_auroc_fold_{int(row.fold) + 1}"] = row.auroc
    return panel


def summary_ends(value):
    """The ends of the fold summary: mean ± SD, or the 95% interval."""
    if hasattr(value, "auroc_sd"):
        return value.auroc_mean - value.auroc_sd, value.auroc_mean + value.auroc_sd
    return value.auroc_ci_low, value.auroc_ci_high


def clipped_errorbar(mean, untrimmed_low, untrimmed_high):
    """Return AUROC-scale error lengths and the ends clipped to that scale."""
    low, high = AUROC_LIMITS
    if not low <= mean <= high:
        raise ValueError(f"AUROC mean {mean:.3f} falls outside {low:g}–{high:g}")
    drawn_low = max(low, untrimmed_low)
    drawn_high = min(high, untrimmed_high)
    cuts = []
    if untrimmed_low < low:
        cuts.append("left")
    if untrimmed_high > high:
        cuts.append("right")
    return np.array([[mean - drawn_low], [drawn_high - mean]]), drawn_low, drawn_high, cuts


def draw(grade_summary, grade_folds, overall_folds):
    """Draw horizontal AUROC dot plots in the panel order of the scores-by-grade figure."""
    fig = plt.figure(figsize=style.figure_size(style.FULL_WIDTH_MM, HEIGHT_MM))
    grid = fig.add_gridspec(2, 4, left=0.165, right=0.985, bottom=0.16, top=0.92,
                            wspace=0.34, hspace=0.48)
    source = {}
    score_axes = []
    grade_positions = np.arange(len(POSITIVE_GRADES), 0, -1)
    lung_findings = list(FINDINGS["lung"].items())
    liver_findings = list(FINDINGS["liver"].items())
    panel_order = [
        [("lung", finding, title) for finding, title in lung_findings[:4]],
        [("lung", *lung_findings[4]), *(("liver", finding, title)
                                           for finding, title in liver_findings), None],
    ]
    for row, panels in enumerate(panel_order):
        for column, panel_spec in enumerate(panels):
            if panel_spec is None:
                continue
            organ, finding, title = panel_spec
            panel_grid = grid[row, column].subgridspec(1, 2, width_ratios=[5, 1.4], wspace=0.12)
            ax = fig.add_subplot(panel_grid[0], sharey=score_axes[0] if score_axes else None)
            count_ax = fig.add_subplot(panel_grid[1], sharey=ax)
            score_axes.append(ax)
            panel = source_panel(grade_summary, grade_folds, overall_folds, organ, finding)
            for position, value in zip(grade_positions, panel.itertuples(index=False)):
                xerr, drawn_low, drawn_high, clipped_ends = clipped_errorbar(
                    value.auroc_mean, *summary_ends(value))
                selector = panel["grade_group"] == value.grade_group
                panel.loc[selector, "errorbar_low_drawn"] = drawn_low
                panel.loc[selector, "errorbar_high_drawn"] = drawn_high
                panel.loc[selector, "errorbar_clipped_ends"] = ", ".join(clipped_ends)
                color = (style.REFERENCE_A if value.grade_group == "not graded"
                         else GRADE_COLORS[value.grade_group])
                ax.errorbar(value.auroc_mean, position, xerr=xerr, fmt="o", markersize=5,
                            color=color,
                            markerfacecolor="white" if value.grade_group == "not graded" else color,
                            markeredgecolor=color, markeredgewidth=style.LINE_WIDTH,
                            elinewidth=style.LINE_WIDTH, capsize=3, capthick=style.LINE_WIDTH,
                            zorder=3, clip_on=False)
            overall_mean = panel["overall_auroc_mean"].iloc[0]
            ax.axvline(overall_mean, color=style.BLACK, linestyle="--", linewidth=style.LINE_WIDTH,
                       zorder=1)
            ax.set_title(title.replace(" of ", " of\n"))
            ax.set_xlim(*AUROC_LIMITS)
            ax.set_xticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1"])
            ax.set_ylim(0.45, len(POSITIVE_GRADES) + 0.55)
            ax.set_yticks(grade_positions, [GRADE_LABELS[group] for group in POSITIVE_GRADES])
            if column:
                ax.tick_params(axis="y", left=False, labelleft=False)
            if row == 1:
                ax.set_xlabel("AUROC against\nreport-negative slides")
            else:
                ax.tick_params(axis="x", labelbottom=False)

            count_ax.set_xlim(0, 1)
            count_ax.set_xticks([])
            count_ax.set_yticks(grade_positions)
            count_ax.tick_params(axis="y", left=False, labelleft=False)
            for spine in count_ax.spines.values():
                spine.set_visible(False)
            count_ax.set_title("n", loc="right")
            n_folds = overall_folds.loc[overall_folds["organ_group"] == organ, "fold"].nunique()
            for position, value in zip(grade_positions, panel.itertuples(index=False)):
                dagger = "†" if value.n_folds < n_folds else ""
                count_ax.text(0.98, position, f"{int(value.grade_slides):,}{dagger}",
                              ha="right", va="center")
            source[f"{organ}_{finding}"] = panel
    return fig, source, score_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--reference-slides", type=Path, required=True)
    ap.add_argument("--folds-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--summary", choices=["sd", "t"], default="sd")
    ap.add_argument("--draft", action="store_true",
                    help="render from predictions without a complete inference record; "
                         "the provenance records the display as a draft")
    args = ap.parse_args()

    values = evaluation_data(args.predictions, args.reference_slides, None, args.folds_dir,
                             args.summary, include_baseline=False, draft=args.draft)
    grade_summary, grade_folds = per_grade_metrics(values["scored"], summary=args.summary)
    style.apply()
    figure, source, score_axes = draw(grade_summary, grade_folds, values["fold_metrics"])
    inputs = display_inputs(args.predictions, args.reference_slides, folds_dir=args.folds_dir,
                            draft=args.draft)
    save_figure(figure, args.out_dir, "supplementary_figure_discrimination_by_grade", source,
                inputs, draft=args.draft, data_axes=score_axes)


if __name__ == "__main__":
    main()

