"""Paired comparisons of modelling choices and learning-curve summaries.

Each comparison family (aggregators, encoders or training fractions) is a set
of five-fold test predictions with one extra column, ``variant``: the
aggregator, the encoder or the training fraction of the run. Absolute metrics
keep the outer-fold structure (mean and SD over folds, as in the primary
analysis). A difference between a variant and the reference configuration is
estimated on the pooled held-out predictions of the slides scored by both,
with a case-clustered percentile bootstrap that resamples the same cases for
both configurations and for every finding, so that a macro difference has an
interval too.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from champs_pipeline.eval.discrimination import (macro_fold_metrics, metric_value,
                                                 per_fold_metrics, summarise_metrics)
from champs_pipeline.eval.predictions import check_test_folds
from champs_pipeline.figures.labels import FINDINGS


METRICS = ("auroc", "average_precision")
# The run axis that distinguishes the variants of each comparison family.
VARIANT_AXES = {
    "aggregator_comparison": "aggregator",
    "encoder_comparison": "encoder",
    "learning_curve": "train_fraction",
}
# The variant of the reference configuration, CLAM-MB on Virchow2 features.
REFERENCES = {
    "aggregator_comparison": "clam_mb",
    "encoder_comparison": "virchow2",
}
MODEL_AXES = ["encoder", "aggregator", "train_fraction"]
VARIANT_KEY = ["variant", "organ_group", "slide_id", "finding"]


def _in_family(frame, family):
    """Mark the rows of runs that belong to ``family``."""
    return frame["families"].str.split(";").map(lambda families: family in families)


def family_predictions(predictions, family, folds_dir):
    """Return the five-fold test predictions of one comparison family with a ``variant`` column.

    The headline runs of the family's organs carry the reference configuration
    and join every family; of the families, only the encoder family does not
    list them itself. The selection must hold one model per variant, organ,
    slide and finding, vary only along the family's axis and cover the
    complete fold set of every variant.
    """
    axis = VARIANT_AXES[family]
    test = predictions.loc[predictions["split"].eq("test")
                           & predictions["design"].eq("fivefold")
                           & predictions["label_variant"].eq("elig")]
    members = _in_family(test, family)
    organs = test.loc[members, "organ_group"].unique()
    headline = _in_family(test, "headline") & test["organ_group"].isin(organs)
    selected = test.loc[members | headline].copy()
    if selected.empty:
        raise ValueError(f"no five-fold test predictions for {family}")
    selected["variant"] = selected[axis]
    fixed = [name for name in MODEL_AXES if name != axis]
    if selected.groupby("organ_group")[fixed].nunique().gt(1).any().any():
        raise ValueError(f"{family} varies along more than its {axis} axis")
    if selected.duplicated(VARIANT_KEY).any():
        raise ValueError(f"{family} mixes several models of one variant for the same test slide")
    for _, rows in selected.groupby("variant"):
        check_test_folds(rows, folds_dir)
    return selected


def variant_summaries(scored, summary="sd"):
    """Return mean and SD over folds of AUROC and AP for each variant and finding."""
    frames = []
    for variant, group in scored.groupby("variant", sort=False):
        table, _, _ = summarise_metrics(group, summary=summary)
        table.insert(0, "variant", variant)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def _interval(samples):
    """Return the 95% percentile interval of bootstrap replicates."""
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _paired_slides(reference, variant, findings):
    """Return, for every finding, the slides scored by both configurations."""
    paired = {}
    for finding in findings:
        left = reference.loc[reference["finding"] == finding,
                             ["slide_id", "case_id", "label", "score"]]
        right = variant.loc[variant["finding"] == finding, ["slide_id", "score"]]
        paired[finding] = left.merge(right, on="slide_id", suffixes=("_reference", "_variant"),
                                     validate="one_to_one")
    return paired


def _case_positions(paired):
    """Number the cases of the paired slides in sorted order."""
    cases = sorted(set(pd.concat([frame["case_id"] for frame in paired.values()])))
    return {case: position for position, case in enumerate(cases)}


def _finding_arrays(frame, case_positions):
    """Return the labels, both scores and the case position of every paired slide."""
    return (frame["label"].to_numpy(int),
            frame["score_reference"].to_numpy(float),
            frame["score_variant"].to_numpy(float),
            frame["case_id"].map(case_positions).to_numpy(int))


def _point_estimates(arrays):
    """Return both configurations' metrics for every finding and for the macro average."""
    point = {}
    for finding, (labels, reference_scores, variant_scores, _) in arrays.items():
        for metric in METRICS:
            point[(finding, metric, "reference")] = metric_value(labels, reference_scores, metric)
            point[(finding, metric, "variant")] = metric_value(labels, variant_scores, metric)
    for metric in METRICS:
        for side in ("reference", "variant"):
            values = [point[(finding, metric, side)] for finding in arrays]
            point[("macro", metric, side)] = float(np.mean(values))
    return point


def _replicate_differences(arrays, case_weights):
    """Return the variant minus reference metric of every finding under one case draw."""
    values = {}
    for finding, (labels, reference_scores, variant_scores, cases) in arrays.items():
        weights = case_weights[cases]
        for metric in METRICS:
            variant = metric_value(labels, variant_scores, metric, weights)
            reference = metric_value(labels, reference_scores, metric, weights)
            values[(finding, metric)] = variant - reference
    return values


def _bootstrap_differences(arrays, n_cases, n_bootstrap, generator):
    """Return the defined replicates of every finding's difference and of the macro difference.

    A draw in which a finding lacks one class is discarded; at most twenty
    draws are made per requested replicate.
    """
    findings = list(arrays)
    replicates = {(finding, metric): [] for finding in [*findings, "macro"] for metric in METRICS}
    attempts = 0
    while len(replicates[("macro", "auroc")]) < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        drawn = generator.integers(0, n_cases, size=n_cases)
        case_weights = np.bincount(drawn, minlength=n_cases).astype(float)
        values = _replicate_differences(arrays, case_weights)
        if not all(np.isfinite(value) for value in values.values()):
            continue
        for key, value in values.items():
            replicates[key].append(value)
        for metric in METRICS:
            macro = np.mean([values[(finding, metric)] for finding in findings])
            replicates[("macro", metric)].append(float(macro))
    if len(replicates[("macro", "auroc")]) < n_bootstrap:
        raise ValueError("too few defined paired bootstrap replicates")
    return replicates


def _metric_columns(point, replicates, finding, metric):
    """Return both estimates, their difference and its interval for one finding and metric."""
    reference = point[(finding, metric, "reference")]
    variant = point[(finding, metric, "variant")]
    low, high = _interval(replicates[(finding, metric)])
    return {f"reference_{metric}": reference,
            f"variant_{metric}": variant,
            f"difference_{metric}": variant - reference,
            f"difference_{metric}_ci_low": low,
            f"difference_{metric}_ci_high": high}


def _paired_rows(paired, n_bootstrap, generator):
    """Return one row of paired estimates for every finding and for the macro average."""
    case_positions = _case_positions(paired)
    arrays = {finding: _finding_arrays(frame, case_positions)
              for finding, frame in paired.items()}
    point = _point_estimates(arrays)
    replicates = _bootstrap_differences(arrays, len(case_positions), n_bootstrap, generator)
    slides = {finding: len(frame) for finding, frame in paired.items()}
    slides["macro"] = int(np.mean(list(slides.values())))
    rows = []
    for finding in [*paired, "macro"]:
        row = {"finding": finding, "slides_paired": slides[finding],
               "cases": len(case_positions)}
        for metric in METRICS:
            row.update(_metric_columns(point, replicates, finding, metric))
        rows.append(row)
    return rows


def paired_differences(scored, reference_variant, n_bootstrap=1000, seed=20260916):
    """Return pooled estimates and paired case-bootstrap intervals against a reference.

    For each organ and variant, the slides scored by both the variant and the
    reference are paired for every finding. Each bootstrap replicate draws
    cases of the organ with replacement and reweights every finding's paired
    slides by the number of times their case was drawn; the same draw serves
    the reference, the variant, every finding and the macro average.
    """
    generator = np.random.default_rng(seed)
    rows = []
    for organ, organ_scores in scored.groupby("organ_group", sort=True):
        findings = list(FINDINGS[organ])
        reference = organ_scores[organ_scores["variant"] == reference_variant]
        if reference.empty:
            raise ValueError(f"{organ}: reference variant {reference_variant!r} not found")
        for variant, variant_scores in organ_scores.groupby("variant", sort=False):
            if variant == reference_variant:
                continue
            paired = _paired_slides(reference, variant_scores, findings)
            unpaired = [finding for finding, frame in paired.items() if frame.empty]
            if unpaired:
                raise ValueError(f"{organ} {variant}: no paired slides for {unpaired}")
            comparison = {"organ_group": organ, "variant": variant,
                          "reference": reference_variant}
            for row in _paired_rows(paired, n_bootstrap, generator):
                rows.append({**comparison, **row})
    return pd.DataFrame(rows)


def counts_path(runs_dir, run_id):
    """Return the training-set counts file that a run writes."""
    return Path(runs_dir) / run_id / "data_counts.json"


def training_counts(run_ids, runs_dir):
    """Return the training cases, slides and positive slides of every run and finding."""
    rows = []
    for run_id in run_ids:
        train = json.loads(counts_path(runs_dir, run_id).read_text())["train"]
        for finding, labels in train["labels"].items():
            rows.append({"run_id": run_id, "finding": finding,
                         "training_cases": train["cases"],
                         "training_slides": train["slides"],
                         "training_positives": labels["pos"]})
    return pd.DataFrame(rows)


def learning_curve_summary(scored, counts, summary="sd"):
    """Return metrics by organ, training fraction and finding with the training-set size.

    ``counts`` holds one row per run and finding, as returned by
    ``training_counts``. Training cases and slides are averaged over the folds
    of an organ and fraction, and training positives over the folds of an
    organ, fraction and finding.
    """
    runs = scored[["run_id", "organ_group", "variant"]].drop_duplicates()
    missing = set(runs["run_id"]) - set(counts["run_id"])
    if missing:
        raise ValueError(f"{len(missing)} learning-curve runs have no training counts")
    runs = runs.rename(columns={"variant": "fraction"})
    counts = counts.merge(runs, on="run_id", validate="many_to_one")
    sizes = (counts.drop_duplicates("run_id")
             .groupby(["organ_group", "fraction"])[["training_cases", "training_slides"]]
             .mean().reset_index())
    positives = (counts.groupby(["organ_group", "fraction", "finding"])["training_positives"]
                 .mean().reset_index())
    table = variant_summaries(scored, summary=summary).rename(columns={"variant": "fraction"})
    table = table.merge(sizes, on=["organ_group", "fraction"], how="left",
                        validate="many_to_one")
    table = table.merge(positives, on=["organ_group", "fraction", "finding"], how="left",
                        validate="many_to_one")
    findings = table["finding"] != "macro"
    if table.loc[findings, "training_positives"].isna().any():
        raise ValueError("a finding has no recorded training positives")
    return table.sort_values(["organ_group", "finding", "fraction"]).reset_index(drop=True)


def learning_curve_folds(scored):
    """Return per-fold macro and per-finding metrics at every training fraction."""
    frames = []
    for fraction, group in scored.groupby("variant", sort=False):
        folds = pd.concat([per_fold_metrics(group), macro_fold_metrics(group)], ignore_index=True)
        folds.insert(0, "fraction", fraction)
        frames.append(folds)
    return pd.concat(frames, ignore_index=True)
