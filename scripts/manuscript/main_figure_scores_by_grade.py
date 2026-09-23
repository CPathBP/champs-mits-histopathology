"""Render model-score distributions by report-negative status and Reference A grade."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from champs_pipeline.eval.discrimination import GRADE_LABELS, GRADE_ORDER, grade_groups
from champs_pipeline.eval.display_data import display_inputs, evaluation_data
from champs_pipeline.figures import style
from champs_pipeline.figures.labels import FINDINGS
from champs_pipeline.figures.output import save_figure


HEIGHT_MM = 100


GRADE_COLORS = dict(zip(GRADE_ORDER[1:-1], style.ordered_colors(4, "Oranges")))


NEARLY_IDENTICAL_RANGE = 0.005


def draw_group(ax, position, values, group, generator):
    """Draw one severity group as a violin or, for sparse groups, individual slides."""
    color = style.GREY if group == "report-negative" else GRADE_COLORS.get(group, style.REFERENCE_A)
    # A violin with a range smaller than about 0.5% of the score scale has no
    # visible body at print size.  Points make those groups visible instead.
    if len(values) < 10 or np.ptp(values) <= NEARLY_IDENTICAL_RANGE:
        if len(values):
            jitter = generator.uniform(-0.20, 0.20, len(values))
            if group == "not graded":
                ax.scatter(values, position + jitter, s=12, facecolors="white",
                           edgecolors=style.REFERENCE_A, linewidths=style.LINE_WIDTH, zorder=3)
            else:
                ax.scatter(values, position + jitter, s=12, color=color, linewidths=0, zorder=3)
        return
    artists = ax.violinplot(values, positions=[position], widths=0.82, vert=False,
                            showextrema=False, showmedians=True)
    body = artists["bodies"][0]
    body.set_linewidth(style.LINE_WIDTH)
    if group == "not graded":
        body.set_facecolor("none")
        body.set_edgecolor(style.REFERENCE_A)
    else:
        body.set_facecolor(color)
        body.set_edgecolor(color)
    body.set_alpha(1)
    artists["cmedians"].set_color(style.BLACK)
    artists["cmedians"].set_linewidth(style.LINE_WIDTH)


def draw(scored):
    """Draw the seven finding panels in the established organ and finding order."""
    fig = plt.figure(figsize=style.figure_size(style.FULL_WIDTH_MM, HEIGHT_MM))
    # Each panel has a score axis and a slim, separate count axis.  Keeping the
    # latter outside the 0--1 data range makes high scores readable as well.
    grid = fig.add_gridspec(2, 4, left=0.165, right=0.985, bottom=0.10, top=0.92,
                            wspace=0.34, hspace=0.48)
    grouped = grade_groups(scored)
    generator = np.random.default_rng(20260914)
    source = {}
    grade_positions = np.arange(len(GRADE_ORDER), 0, -1)
    score_axes = []
    lung_findings = list(FINDINGS["lung"].items())
    liver_findings = list(FINDINGS["liver"].items())
    panel_order = [
        [("lung", finding, title) for finding, title in lung_findings[:4]],
        [("lung", *lung_findings[4]),
         *(("liver", finding, title) for finding, title in liver_findings),
         None],
    ]
    for row, panels in enumerate(panel_order):
        for column, panel_spec in enumerate(panels):
            if panel_spec is None:
                continue
            organ, finding, title = panel_spec
            # The dedicated count strip is deliberately wide enough for four-digit
            # counts, with a print-visible gap from the 0--1 score axis.
            panel_grid = grid[row, column].subgridspec(1, 2, width_ratios=[5, 1.4], wspace=0.12)
            ax = fig.add_subplot(panel_grid[0], sharey=score_axes[0] if score_axes else None)
            count_ax = fig.add_subplot(panel_grid[1], sharey=ax)
            score_axes.append(ax)
            selected = (grouped["organ_group"] == organ) & (grouped["finding"] == finding)
            panel = grouped[selected].copy()
            source[f"{organ}_{finding}"] = panel
            for position, group in zip(grade_positions, GRADE_ORDER):
                values = panel.loc[panel["grade_group"] == group, "score"].to_numpy()
                draw_group(ax, position, values, group, generator)
            # The two aspiration names need two lines at this panel width; the
            # rendered wording remains the finding name and leaves room for n.
            ax.set_title(title.replace(" of ", " of\n"))
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_ylim(0.45, len(GRADE_ORDER) + 0.55)
            ax.set_yticks(grade_positions, [GRADE_LABELS[group] for group in GRADE_ORDER])
            if column:
                ax.tick_params(axis="y", left=False, labelleft=False)
            if row == 1:
                ax.set_xlabel("Model score")
            else:
                ax.tick_params(axis="x", labelbottom=False)

            count_ax.set_xlim(0, 1)
            count_ax.set_xticks([])
            count_ax.set_yticks(grade_positions)
            count_ax.tick_params(axis="y", left=False, labelleft=False)
            for spine in count_ax.spines.values():
                spine.set_visible(False)
            count_ax.set_title("n", loc="right")
            for position, group in zip(grade_positions, GRADE_ORDER):
                count_ax.text(0.98, position, f"{(panel['grade_group'] == group).sum():,}",
                              ha="right", va="center")
    return fig, source, score_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--reference-slides", type=Path, required=True)
    ap.add_argument("--folds-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--draft", action="store_true",
                    help="render from predictions without a complete inference record; "
                         "the provenance records the display as a draft")
    args = ap.parse_args()

    values = evaluation_data(args.predictions, args.reference_slides, None, args.folds_dir,
                             include_baseline=False, draft=args.draft)
    style.apply()
    figure, source, score_axes = draw(values["scored"])
    inputs = display_inputs(args.predictions, args.reference_slides, folds_dir=args.folds_dir,
                            draft=args.draft)
    save_figure(figure, args.out_dir, "main_figure_scores_by_grade", source, inputs,
                draft=args.draft, data_axes=score_axes, clear_axes=True)


if __name__ == "__main__":
    main()

