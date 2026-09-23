"""Render five-fold discrimination within sites and within year-of-death bands."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from champs_pipeline.eval.display_data import display_inputs, evaluation_data
from champs_pipeline.eval.strata import (YEAR_BANDS, adjusted_auroc, attach_strata, slide_strata,
                                         stratum_metrics)
from champs_pipeline.figures import style
from champs_pipeline.figures.labels import SITES
from champs_pipeline.figures.output import save_figure
from champs_pipeline.figures.strata_panels import add_legend, axis_floor, draw_row, pooled_auroc


HEIGHT_MM = 130
# For each stratum: the grid rows of its panels, the levels drawn and the letter of its
# first panel. Grid row 2 is a spacer that keeps the tick labels of the sites clear of the
# titles below them.
ROWS = {
    "site": ((0, 1), list(SITES), "a"),
    "year_of_death": ((3, 4), YEAR_BANDS, "h"),
}
SUMMARY = ["organ_group", "finding", "stratum", "pooled_auroc", "adjusted_auroc",
           "prevalence_auroc"]


def figure(metrics, adjusted):
    """AUROC within sites and within year bands, one column per finding."""
    fig = plt.figure(figsize=style.figure_size(style.FULL_WIDTH_MM, HEIGHT_MM))
    grid = fig.add_gridspec(5, 7, height_ratios=[3.2, 1, 1.1, 3.2, 1], left=0.062,
                            right=0.995, bottom=0.22, top=0.92, wspace=0.16, hspace=0.2)
    floor = axis_floor(metrics)
    for stratum, (grid_rows, levels, letter) in ROWS.items():
        block = metrics[metrics["stratum"] == stratum]
        pooled = pooled_auroc(adjusted, stratum)
        draw_row(fig, grid, grid_rows, block, pooled, levels, levels, floor, letter)
    add_legend(fig, y=0.012)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--reference-slides", type=Path, required=True)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--folds-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000, help="case-bootstrap replicates")
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--draft", action="store_true",
                    help="render from predictions without a complete inference record; "
                         "the provenance records the display as a draft")
    args = ap.parse_args()

    data = evaluation_data(args.predictions, args.reference_slides, args.cohort_dir,
                           args.folds_dir, include_baseline=False, draft=args.draft)
    cohort_slides = pd.read_csv(args.cohort_dir / "cohort_slides.csv", low_memory=False)
    cohort_cases = pd.read_csv(args.cohort_dir / "cohort_cases.csv", low_memory=False)
    strata, _ = slide_strata(cohort_slides, cohort_cases)
    scored = attach_strata(data["scored"], strata)
    options = {"n_bootstrap": args.bootstrap, "seed": args.seed}
    metrics = [stratum_metrics(scored, stratum, **options) for stratum in ROWS]
    adjusted = [adjusted_auroc(scored, stratum, **options) for stratum in ROWS]
    metrics = pd.concat(metrics, ignore_index=True)
    adjusted = pd.concat(adjusted, ignore_index=True)

    inputs = display_inputs(args.predictions, args.reference_slides, args.cohort_dir,
                            args.folds_dir, draft=args.draft)
    source = metrics.merge(adjusted[SUMMARY], on=["organ_group", "finding", "stratum"])
    parts = {
        "a": source[source["stratum"] == "site"],
        "h": source[source["stratum"] == "year_of_death"],
    }
    style.apply()
    save_figure(figure(metrics, adjusted), args.out_dir, "supplementary_figure_strata", parts,
                inputs, draft=args.draft)


if __name__ == "__main__":
    main()
