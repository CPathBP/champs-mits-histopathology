"""Discrimination within strata of patient, specimen and acquisition context.

Every stratum level is scored on the pooled held-out predictions of the slides
in that level, so a level shows whether the score separates positive from
negative slides inside it. Intervals are case-clustered percentile bootstraps.
The summary over the levels of one stratum is the covariate-adjusted AUROC of
Janes and Pepe: the within-level AUROC values averaged with the share of
positive slides in each level as weight, which is the AUROC of slide pairs
drawn within levels.
"""

from functools import partial

import numpy as np
import pandas as pd

from champs_pipeline.eval.discrimination import metric_value
from champs_pipeline.figures.labels import SITES, SLIDE_SOURCES, site_code


NOT_RECORDED = "Not recorded"

# Age groups as recorded in the cohort tables, with a short label for figure ticks.
AGE_GROUPS = {
    "Stillbirth": "SB",
    "Death in the first 24 hours": "<1 d",
    "Early Neonate (1 to 6 days)": "1–6 d",
    "Late Neonate (7 to 27 days)": "7–27 d",
    "Infant (28 days to less than 12 months)": "1–11 mo",
    "Child (12 months to less than 60 Months)": "1–4 y",
}
TISSUE_TERTILES = ["Lowest tertile", "Middle tertile", "Highest tertile"]
POSTMORTEM_BANDS = ["≤6 h", ">6 to 24 h", ">24 h"]
POSTMORTEM_EDGES = [-np.inf, 6, 24, np.inf]
YEAR_BANDS = ["2016–2019", "2020–2021", "2022–2024"]
YEAR_EDGES = [2015, 2019, 2021, 2024]

# Stratum column, display name and level order. ``Not recorded`` is a level
# where the covariate can be missing.
STRATA = {
    "age_group": ("Age group", [*AGE_GROUPS, NOT_RECORDED]),
    "scan_source": ("Scan source", [SLIDE_SOURCES["CPL"], SLIDE_SOURCES["SITE"]]),
    "tissue": ("Tissue amount", TISSUE_TERTILES),
    "postmortem_interval": ("Post-mortem interval", [*POSTMORTEM_BANDS, NOT_RECORDED]),
    "site": ("Site", list(SITES)),
    "year_of_death": ("Year of death", [*YEAR_BANDS, NOT_RECORDED]),
}
# A tuple, not a set: the order fixes the columns and the sequence of bootstrap draws.
METRICS = ("auroc", "average_precision")
ADJUSTED_ESTIMATES = ("pooled_auroc", "adjusted_auroc", "difference_auroc", "prevalence_auroc")
SLIDE_COLUMNS = ["slide_id", "champs_deid", "organ_group", "site", "slide_source",
                 "n_patches_virchow2"]
CASE_COLUMNS = ["champs_deid", "age_group", "calc_postmortem_hrs", "death_year"]


def slide_strata(cohort_slides, cohort_cases):
    """Return one row per training-cohort slide with its stratum levels.

    Tissue amount is the tertile of the Virchow2 tile count among the
    training-cohort slides of the same organ; the cut points of each organ are
    returned beside the frame.
    """
    slides = cohort_slides.loc[cohort_slides["training"].astype(bool), SLIDE_COLUMNS]
    cases = cohort_cases.loc[:, CASE_COLUMNS]
    frame = slides.merge(cases, on="champs_deid", how="left", validate="many_to_one")
    frame = frame.rename(columns={"champs_deid": "case_id"})
    frame["age_group"] = frame["age_group"].fillna(NOT_RECORDED)
    frame["scan_source"] = frame["slide_source"].map(SLIDE_SOURCES)
    frame["site"] = frame["site"].map(site_code)
    tissue, cut_points = _tissue_tertiles(frame)
    frame["tissue"] = tissue
    frame["postmortem_interval"] = _bands(frame["calc_postmortem_hrs"], POSTMORTEM_EDGES,
                                          POSTMORTEM_BANDS)
    frame["year_of_death"] = _bands(frame["death_year"], YEAR_EDGES, YEAR_BANDS)
    _check_levels(frame)
    return frame.loc[:, ["slide_id", "case_id", "organ_group", *STRATA]], cut_points


def _tissue_tertiles(frame):
    tissue = pd.Series(index=frame.index, dtype=object)
    cut_points = {}
    for organ, group in frame.groupby("organ_group"):
        tertiles, edges = pd.qcut(group["n_patches_virchow2"], 3, labels=TISSUE_TERTILES,
                                  retbins=True)
        tissue.loc[group.index] = tertiles.astype(str)
        cut_points[organ] = [float(edge) for edge in edges]
    return tissue, cut_points


def _bands(values, edges, labels):
    """Cut values into labelled bands closed on the right; a missing value is not recorded."""
    bands = pd.cut(values, edges, labels=labels).astype(object)
    return bands.where(values.notna(), NOT_RECORDED)


def _check_levels(frame):
    for stratum, (_, levels) in STRATA.items():
        outside = ~frame[stratum].isin(levels)
        if outside.any():
            values = sorted(frame.loc[outside, stratum].astype(str).unique())
            raise ValueError(f"{stratum} levels outside the stratum definition: {values}")


def attach_strata(scored, strata):
    """Join the stratum levels to scored slides.

    The scored rows keep their own case. A slide that belongs to another case
    in the cohort tables, or a scored slide without strata, is an error.
    """
    _check_cases(scored, strata)
    levels = strata.drop(columns=["case_id", "organ_group"])
    joined = scored.merge(levels, on="slide_id", how="left", validate="many_to_one")
    unmatched = joined[list(STRATA)].isna().any(axis=1)
    if unmatched.any():
        missing = joined.loc[unmatched, "slide_id"].nunique()
        raise ValueError(f"{missing} scored slides have no strata in the cohort tables")
    return joined


def _check_cases(scored, strata):
    pairs = scored[["slide_id", "case_id"]].drop_duplicates()
    both = pairs.merge(strata[["slide_id", "case_id"]], on="slide_id", suffixes=("", "_cohort"))
    differs = both["case_id"].ne(both["case_id_cohort"])
    if differs.any():
        slides = both.loc[differs, "slide_id"].nunique()
        raise ValueError(f"{slides} scored slides belong to another case in the cohort tables")


def _case_weights(case_index, n_cases, generator):
    """Weight every slide by the number of times its case is drawn."""
    draws = generator.integers(0, n_cases, n_cases)
    return np.bincount(draws, minlength=n_cases)[case_index].astype(float)


def _bootstrap(statistic, case_ids, n_bootstrap, generator, what):
    """Return ``n_bootstrap`` case-bootstrap replicates of ``statistic(weights)``.

    A draw for which the statistic is undefined is replaced; at most twenty
    draws are made per replicate.
    """
    case_index, cases = pd.factorize(case_ids)
    samples = []
    attempts = 0
    while len(samples) < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        values = statistic(_case_weights(case_index, len(cases), generator))
        if not np.isnan(values).any():
            samples.append(values)
    if len(samples) < n_bootstrap:
        raise ValueError(f"only {len(samples)} defined bootstrap replicates for {what}")
    return np.asarray(samples)


def _estimate(name, point, samples):
    """The point estimate with its 95% percentile interval."""
    low, high = np.percentile(samples, [2.5, 97.5])
    return {name: point, f"{name}_ci_low": float(low), f"{name}_ci_high": float(high)}


def _not_estimable(name):
    return dict.fromkeys([name, f"{name}_ci_low", f"{name}_ci_high"], np.nan)


def stratum_metrics(scored, stratum, score_column="score", n_bootstrap=1000, seed=20260916,
                    min_class_count=10):
    """Return AUROC and AP with intervals for every finding and level of one stratum.

    A level with fewer than ``min_class_count`` positive or negative slides
    keeps its counts and is marked not estimable.
    """
    generator = np.random.default_rng(seed)
    rows = []
    groups = scored.groupby(["organ_group", "finding", stratum], sort=True)
    for (organ, finding, level), group in groups:
        labels = group["label"].to_numpy()
        scores = group[score_column].to_numpy(dtype=float)
        positives = int(labels.sum())
        negatives = len(labels) - positives
        row = {
            "organ_group": organ,
            "finding": finding,
            "stratum": stratum,
            "level": level,
            "slides_scored": len(labels),
            "cases": group["case_id"].nunique(),
            "positive_slides": positives,
            "prevalence": positives / len(labels),
            "estimable": min(positives, negatives) >= min_class_count,
        }
        for metric in METRICS:
            if not row["estimable"]:
                row.update(_not_estimable(metric))
                continue
            point = metric_value(labels, scores, metric)
            statistic = partial(metric_value, labels, scores, metric)
            samples = _bootstrap(statistic, group["case_id"], n_bootstrap, generator,
                                 f"{finding} {stratum} {level} {metric}")
            row.update(_estimate(metric, point, samples))
        rows.append(row)
    return pd.DataFrame(rows)


def _usable_levels(group, stratum, exclude_levels):
    """Levels with a positive and a negative slide, other than ``exclude_levels``."""
    counts = group.groupby(stratum)["label"].agg(["sum", "count"])
    both_classes = (counts["sum"] > 0) & (counts["sum"] < counts["count"])
    return [level for level in counts.index[both_classes] if level not in exclude_levels]


def _adjusted_estimates(labels, scores, level_index, weights):
    """Pooled, covariate-adjusted, their difference and prevalence-only AUROC for one weighting.

    All four are NaN when a level with weight lacks one class.
    """
    within = []
    positive_weight = []
    prevalence_score = np.zeros(len(labels))
    for level in np.unique(level_index):
        rows = level_index == level
        total = weights[rows].sum()
        if total == 0:
            continue
        positives = (weights[rows] * labels[rows]).sum()
        prevalence_score[rows] = positives / total
        value = metric_value(labels[rows], scores[rows], "auroc", weights[rows])
        if np.isnan(value):
            return np.full(len(ADJUSTED_ESTIMATES), np.nan)
        within.append(value)
        positive_weight.append(positives)
    pooled = metric_value(labels, scores, "auroc", weights)
    adjusted = float(np.average(within, weights=positive_weight))
    prevalence_only = metric_value(labels, prevalence_score, "auroc", weights)
    return np.array([pooled, adjusted, pooled - adjusted, prevalence_only])


def adjusted_auroc(scored, stratum, score_column="score", n_bootstrap=1000, seed=20260916,
                   exclude_levels=(NOT_RECORDED,)):
    """Return pooled against covariate-adjusted AUROC for every finding of one stratum.

    Levels without a positive or without a negative slide cannot contribute a
    within-level AUROC and are left out of every estimate, as are the levels in
    ``exclude_levels``; their slides are counted in ``slides_excluded``. The
    prevalence-only AUROC scores every slide with the Reference A prevalence of
    its level and shows how much of the pooled value the level membership
    alone reproduces.
    """
    generator = np.random.default_rng(seed)
    rows = []
    for (organ, finding), group in scored.groupby(["organ_group", "finding"], sort=True):
        usable = _usable_levels(group, stratum, exclude_levels)
        kept = group[group[stratum].isin(usable)]
        labels = kept["label"].to_numpy()
        scores = kept[score_column].to_numpy(dtype=float)
        level_index = pd.factorize(kept[stratum])[0]
        statistic = partial(_adjusted_estimates, labels, scores, level_index)
        point = statistic(np.ones(len(labels)))
        samples = _bootstrap(statistic, kept["case_id"], n_bootstrap, generator,
                             f"{finding} {stratum}")
        row = {
            "organ_group": organ,
            "finding": finding,
            "stratum": stratum,
            "levels_used": len(usable),
            "levels_total": group[stratum].nunique(),
            "slides_scored": len(kept),
            "slides_excluded": len(group) - len(kept),
            "positive_slides": int(labels.sum()),
        }
        for column, name in enumerate(ADJUSTED_ESTIMATES):
            row.update(_estimate(name, point[column], samples[:, column]))
        rows.append(row)
    return pd.DataFrame(rows)
