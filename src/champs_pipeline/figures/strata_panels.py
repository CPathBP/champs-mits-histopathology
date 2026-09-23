"""Panel row shared by the stratum figures: AUROC per level above a prevalence strip."""

import numpy as np
from matplotlib.patches import Patch

from champs_pipeline.figures import style
from champs_pipeline.figures.labels import study_findings


# Two-line titles fit the seven narrow columns.
TITLES = {
    "aspiration_squames": "Aspiration\nof squames",
    "aspiration_meconium": "Aspiration\nof meconium",
    "bronchopneumonia": "Broncho-\npneumonia",
    "pneumonitis": "Pneumonitis",
    "hyaline_membranes": "Hyaline\nmembranes",
    "steatosis": "Steatosis",
    "hemozoin_pigment": "Hemozoin\npigment",
}
MARKER_SIZE = 3.6
GRID_COLOR = "#E5E5E5"


def pooled_auroc(adjusted, stratum):
    """Map (organ, finding) to the pooled AUROC of one stratum."""
    rows = adjusted[adjusted["stratum"] == stratum]
    return rows.set_index(["organ_group", "finding"])["pooled_auroc"].to_dict()


def axis_floor(metrics, lowest=0.5):
    """Shared lower AUROC limit: at most ``lowest`` and below every interval drawn."""
    low = metrics.loc[metrics["estimable"], "auroc_ci_low"].min()
    return min(lowest, np.floor(low * 20) / 20)


def draw_row(fig, grid, rows, metrics, pooled, levels, tick_labels, floor, first_letter):
    """Draw one stratum: seven AUROC panels with a prevalence strip under each.

    ``rows`` gives the grid rows of the AUROC panels and of the prevalence
    strips; the grid has one column per finding. ``pooled`` maps
    (organ, finding) to the pooled AUROC.
    """
    upper_row, lower_row = rows
    positions = np.arange(len(levels))
    for column, (organ, finding) in enumerate(study_findings()):
        upper = fig.add_subplot(grid[upper_row, column])
        lower = fig.add_subplot(grid[lower_row, column], sharex=upper)
        block = metrics[(metrics["organ_group"] == organ) & (metrics["finding"] == finding)]
        block = block.set_index("level").reindex(levels)
        _auroc_panel(upper, block, pooled[(organ, finding)], positions, floor)
        upper.set_title(TITLES[finding], fontsize=8, fontweight="bold", pad=3)
        _prevalence_strip(lower, block, positions, tick_labels)
        if column == 0:
            upper.set_ylabel("AUROC", fontsize=8, labelpad=2)
            lower.set_ylabel("Positive (%)", fontsize=8, labelpad=2)
        else:
            upper.tick_params(axis="y", labelleft=False)
            lower.tick_params(axis="y", labelleft=False)
        letter = chr(ord(first_letter) + column)
        offset = -16 if column == 0 else -8
        style.panel_label(upper, letter, dx_pt=offset, dy_pt=14)


def _auroc_panel(ax, block, pooled, positions, floor):
    """Estimable levels as points with intervals, and the pooled AUROC as a line."""
    estimable = block["estimable"].eq(True).to_numpy()
    auroc = block["auroc"].to_numpy()[estimable]
    below = auroc - block["auroc_ci_low"].to_numpy()[estimable]
    above = block["auroc_ci_high"].to_numpy()[estimable] - auroc
    ax.axhline(pooled, color=style.GREY, linewidth=style.LINE_WIDTH, zorder=1)
    ax.errorbar(positions[estimable], auroc, yerr=[below, above], fmt="o", color=style.MODEL,
                markersize=MARKER_SIZE, capsize=0, linewidth=style.LINE_WIDTH, zorder=3)
    ax.set_ylim(floor, 1.0)
    ax.set_xlim(-0.6, len(positions) - 0.4)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=style.LINE_WIDTH)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8, length=2)
    ax.tick_params(axis="x", labelbottom=False, bottom=False)


def _prevalence_strip(ax, block, positions, tick_labels):
    """Share of report-positive slides in every level, with the level names as ticks."""
    prevalence = block["prevalence"].fillna(0).to_numpy() * 100
    ax.bar(positions, prevalence, width=0.62, color=style.REFERENCE_A, linewidth=0)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(positions, tick_labels, rotation=90, ha="center", va="top")
    ax.tick_params(axis="both", labelsize=8, length=2)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=style.LINE_WIDTH)
    ax.set_axisbelow(True)


def add_legend(fig, y=0.02):
    """Legend for the within-level AUROC, the pooled line and the prevalence bars."""
    ax = fig.axes[0]
    within = ax.errorbar([], [], yerr=[], fmt="o", color=style.MODEL, markersize=MARKER_SIZE,
                         linewidth=style.LINE_WIDTH, label="Model AUROC (95% CI) within level")
    pooled, = ax.plot([], [], color=style.GREY, linewidth=style.LINE_WIDTH,
                      label="All levels pooled")
    prevalence = Patch(color=style.REFERENCE_A, label="Report-positive slides (%)")
    handles = [within, pooled, prevalence]
    labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, fontsize=8,
               frameon=False)
