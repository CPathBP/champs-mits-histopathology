"""Render discrimination within age groups, and the stratum tables of the supplement."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from champs_pipeline.eval.display_data import display_inputs, evaluation_data
from champs_pipeline.eval.strata import (AGE_GROUPS, STRATA, adjusted_auroc, attach_strata,
                                         slide_strata, stratum_metrics)
from champs_pipeline.figures import style
from champs_pipeline.figures.labels import FINDINGS, study_findings
from champs_pipeline.figures.output import save_figure, write_table
from champs_pipeline.figures.strata_panels import add_legend, axis_floor, draw_row, pooled_auroc


HEIGHT_MM = 66
TABLE_STRATA = ["age_group", "scan_source", "tissue", "postmortem_interval"]
LEVEL_KEYS = ["organ_group", "finding", "stratum", "level"]
FIGURE_SUMMARY = ["organ_group", "finding", "pooled_auroc", "adjusted_auroc", "prevalence_auroc",
                  "slides_excluded"]


def interval(row, metric):
    """Format an estimate with its 95% interval, or state that it is not estimable."""
    if not row.get("estimable", True) or pd.isna(row[metric]):
        return "Not estimable"
    return f"{row[metric]:.3f} ({row[f'{metric}_ci_low']:.3f} to {row[f'{metric}_ci_high']:.3f})"


def signed_interval(row, name):
    """Format a difference with its sign; a value that rounds to zero is shown as +0.000."""
    value = 0.0 if abs(row[name]) < 0.0005 else row[name]
    return f"{value:+.3f} ({row[f'{name}_ci_low']:.3f} to {row[f'{name}_ci_high']:.3f})"


def with_baseline(scored, baseline):
    """Add the context-baseline score of every scored slide."""
    keys = ["organ_group", "finding", "fold", "slide_id"]
    joined = scored.merge(baseline, on=keys, how="left", validate="one_to_one")
    if joined["baseline_score"].isna().any():
        raise ValueError("a scored slide has no context-baseline score")
    return joined


def stratum_results(scored, n_bootstrap, seed):
    """Model and baseline metrics within levels, and the adjusted AUROC, of the table strata."""
    options = {"n_bootstrap": n_bootstrap, "seed": seed}
    model = [stratum_metrics(scored, stratum, **options) for stratum in TABLE_STRATA]
    baseline = [stratum_metrics(scored, stratum, score_column="baseline_score", **options)
                for stratum in TABLE_STRATA]
    adjusted = [adjusted_auroc(scored, stratum, **options) for stratum in TABLE_STRATA]
    model = pd.concat(model, ignore_index=True)
    baseline = pd.concat(baseline, ignore_index=True)
    adjusted = pd.concat(adjusted, ignore_index=True)
    return model, baseline, adjusted


def figure(metrics, adjusted):
    """AUROC within the age groups, one column per finding."""
    fig = plt.figure(figsize=style.figure_size(style.FULL_WIDTH_MM, HEIGHT_MM))
    grid = fig.add_gridspec(2, 7, height_ratios=[3.2, 1], left=0.062, right=0.995,
                            bottom=0.33, top=0.86, wspace=0.16, hspace=0.25)
    pooled = pooled_auroc(adjusted, "age_group")
    draw_row(fig, grid, (0, 1), metrics, pooled, list(AGE_GROUPS), list(AGE_GROUPS.values()),
             axis_floor(metrics), "a")
    add_legend(fig)
    return fig


def level_row(key, model, baseline):
    """One table row: counts, model and baseline discrimination within one level."""
    organ, finding, stratum, level = key
    counts = (f"{int(model['positive_slides']):,}/{int(model['slides_scored']):,} "
              f"({model['prevalence']:.1%})")
    return {
        "Finding": FINDINGS[organ][finding],
        "Stratum": STRATA[stratum][0],
        "Level": level,
        "Cases": f"{int(model['cases']):,}",
        "Positive/scored": counts,
        "AUROC (95% CI)": interval(model, "auroc"),
        "AP (95% CI)": interval(model, "average_precision"),
        "Baseline AUROC (95% CI)": interval(baseline, "auroc"),
    }


def strata_table(model, baseline):
    """One row per finding, stratum and level with scored slides."""
    model = model.set_index(LEVEL_KEYS).sort_index()
    baseline = baseline.set_index(LEVEL_KEYS).sort_index()
    rows = []
    for organ, finding in study_findings():
        for stratum in TABLE_STRATA:
            for level in STRATA[stratum][1]:
                key = (organ, finding, stratum, level)
                if key in model.index:
                    rows.append(level_row(key, model.loc[key], baseline.loc[key]))
    return pd.DataFrame(rows)


def adjusted_table(adjusted):
    """One row per finding and stratum: pooled against covariate-adjusted AUROC."""
    adjusted = adjusted.set_index(["organ_group", "finding", "stratum"]).sort_index()
    rows = []
    for organ, finding in study_findings():
        for stratum in TABLE_STRATA:
            row = adjusted.loc[(organ, finding, stratum)]
            rows.append({
                "Finding": FINDINGS[organ][finding],
                "Stratum": STRATA[stratum][0],
                "Levels used": f"{int(row['levels_used'])} of {int(row['levels_total'])}",
                "Slides": f"{int(row['slides_scored']):,}",
                "Pooled AUROC (95% CI)": interval(row, "pooled_auroc"),
                "Adjusted AUROC (95% CI)": interval(row, "adjusted_auroc"),
                "Difference (95% CI)": signed_interval(row, "difference_auroc"),
                "Prevalence-only AUROC (95% CI)": interval(row, "prevalence_auroc"),
            })
    return pd.DataFrame(rows)


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
                           args.folds_dir, draft=args.draft)
    cohort_slides = pd.read_csv(args.cohort_dir / "cohort_slides.csv", low_memory=False)
    cohort_cases = pd.read_csv(args.cohort_dir / "cohort_cases.csv", low_memory=False)
    strata, _ = slide_strata(cohort_slides, cohort_cases)
    scored = attach_strata(data["scored"], strata)
    scored = with_baseline(scored, data["baseline"])
    model, baseline, adjusted = stratum_results(scored, args.bootstrap, args.seed)

    inputs = display_inputs(args.predictions, args.reference_slides, args.cohort_dir,
                            args.folds_dir, draft=args.draft)
    age = model[model["stratum"] == "age_group"]
    age_adjusted = adjusted.loc[adjusted["stratum"] == "age_group", FIGURE_SUMMARY]
    source = age.merge(age_adjusted, on=["organ_group", "finding"])
    style.apply()
    save_figure(figure(age, adjusted), args.out_dir, "main_figure_strata", {"": source}, inputs,
                draft=args.draft)
    # The level names are long; wrap them so that the table fits the page width.
    write_table(strata_table(model, baseline), args.out_dir, "supplementary_table_strata",
                inputs, align=r"ll>{\raggedright\arraybackslash}p{2.7cm}rrrrr",
                group_column="Finding", header_width=12, long=True, draft=args.draft)
    write_table(adjusted_table(adjusted), args.out_dir, "supplementary_table_strata_adjusted",
                inputs, group_column="Finding", header_width=12, long=True, draft=args.draft)


if __name__ == "__main__":
    main()
