"""Shared input loading for manuscript discrimination displays."""

from pathlib import Path

import pandas as pd

from champs_pipeline.eval.discrimination import (model_baseline_difference,
                                                        non_image_baseline,
                                                        scored_slides,
                                                        summarise_metrics)
from champs_pipeline.eval import load
from champs_pipeline.eval.predictions import check_test_folds, record_path, select_test_runs


def summary_label(summary):
    """Column label of a fold summary: mean and SD, or a 95% interval."""
    if summary == "sd":
        return "mean ± SD"
    return "95% CI"


def estimate(row, metric, summary):
    """Format the fold summary of one metric as ``mean ± SD`` or ``mean (low to high)``."""
    mean = row[f"{metric}_mean"]
    if summary == "sd":
        return f"{mean:.3f} ± {row[f'{metric}_sd']:.3f}"
    return f"{mean:.3f} ({row[f'{metric}_ci_low']:.3f} to {row[f'{metric}_ci_high']:.3f})"


def evaluation_data(predictions_path, reference_path, cohort_dir, folds_dir, summary="sd",
                    evaluation_variant="elig", include_baseline=True, draft=False):
    """Load one label variant and, when requested, its non-image comparison."""
    predictions = load.predictions(predictions_path).check(draft=draft).frame
    predictions = select_test_runs(predictions, "headline", "fivefold", "elig")
    check_test_folds(predictions, folds_dir)
    reference = pd.read_parquet(reference_path,
                                columns=["organ_group", "slide_id", "finding", "variant", "label",
                                         "severity"])
    scored = scored_slides(predictions, reference, evaluation_variant)
    model_summary, fold_metrics, macro_metrics = summarise_metrics(scored, summary=summary)
    result = {"predictions": predictions, "reference": reference, "scored": scored,
              "model_summary": model_summary, "fold_metrics": fold_metrics,
              "macro_metrics": macro_metrics}
    if include_baseline:
        cases_path = Path(cohort_dir) / "cohort_cases.csv"
        slides_path = Path(cohort_dir) / "cohort_slides.csv"
        cases = pd.read_csv(cases_path, low_memory=False)
        slides = pd.read_csv(slides_path, low_memory=False)
        baseline = non_image_baseline(scored, reference, cases, slides, folds_dir,
                                      evaluation_variant)
        baseline_scored = scored.merge(baseline, on=["organ_group", "finding", "fold", "slide_id"],
                                       how="inner", validate="one_to_one")
        baseline_summary, baseline_fold_metrics, _ = summarise_metrics(
            baseline_scored, score_column="baseline_score", summary=summary)
        difference, difference_folds = model_baseline_difference(scored, baseline, summary)
        result.update({"baseline": baseline, "baseline_summary": baseline_summary,
                       "baseline_fold_metrics": baseline_fold_metrics, "difference": difference,
                       "difference_folds": difference_folds,
                       "cases_path": cases_path, "slides_path": slides_path})
    return result


def display_inputs(predictions_path, reference_path, cohort_dir=None, folds_dir=None,
                   draft=False):
    """Return all material input files used by one display for provenance.

    A draft has no inference record, so the record is not an input.
    """
    paths = [Path(predictions_path)]
    if not draft:
        paths.append(record_path(predictions_path))
    paths.append(Path(reference_path))
    if cohort_dir is not None:
        paths.append(Path(cohort_dir) / "cohort_cases.csv")
        paths.append(Path(cohort_dir) / "cohort_slides.csv")
    if folds_dir is not None:
        paths.extend(sorted(Path(folds_dir).glob("*_elig_fivefold/fold_*.csv")))
    return paths

