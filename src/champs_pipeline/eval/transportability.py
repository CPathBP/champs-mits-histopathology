"""Site-held-out discrimination and paired comparison with five-fold CV.

The input is one row per slide and finding.  It contains scores from both
evaluation designs on the same slide.  Metrics remain slide-level, while all
intervals resample cases and retain every slide belonging to a sampled case.
The pooled paired bootstrap is stratified by site so that it preserves the
observed site composition.  The site-average summary separately gives every
held-out site equal weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from champs_pipeline.eval.discrimination import metric_value


REQUIRED_COLUMNS = {
    "organ_group", "finding", "slide_id", "case_id", "site", "label",
    "fivefold_score", "site_held_out_score",
}
METRICS = ("auroc", "average_precision")


def validate_paired_predictions(frame):
    """Validate and return the paired prediction input in display-ready types."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"paired predictions are missing columns: {sorted(missing)}")
    data = frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("paired predictions contain missing required values")
    keys = ["organ_group", "finding", "slide_id"]
    if data.duplicated(keys).any():
        raise ValueError("paired predictions contain duplicate organ, finding and slide rows")
    data["label"] = data["label"].astype(int)
    if set(data["label"]) - {0, 1}:
        raise ValueError("labels must be binary")
    for column in ("fivefold_score", "site_held_out_score"):
        data[column] = data[column].astype(float)
        if not data[column].between(0, 1).all():
            raise ValueError(f"{column} must lie between zero and one")
    if (data.groupby("slide_id")["site"].nunique() > 1).any():
        raise ValueError("a slide is assigned to more than one site")
    if (data.groupby("case_id")["site"].nunique() > 1).any():
        raise ValueError("a case is assigned to more than one site")
    return data


def _case_indices(group):
    """Return arrays of row indices, one for every case in ``group``."""
    groups = group.groupby("case_id", sort=False).groups
    return [indices.to_numpy() for indices in groups.values()]


def _draw_cases(case_indices, generator):
    selected = generator.integers(0, len(case_indices), len(case_indices))
    return np.concatenate([case_indices[index] for index in selected])


def _interval(samples):
    if not samples:
        return np.nan, np.nan
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_site(group, score_column, metric, n_bootstrap, generator):
    """Case-clustered percentile interval within one held-out site."""
    work = group.reset_index(drop=True)
    cases = _case_indices(work)
    labels = work["label"].to_numpy()
    scores = work[score_column].to_numpy()
    samples = []
    attempts = 0
    while len(samples) < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        selected = _draw_cases(cases, generator)
        value = metric_value(labels[selected], scores[selected], metric)
        if np.isfinite(value):
            samples.append(value)
    if len(samples) < n_bootstrap:
        raise ValueError(f"only {len(samples)} defined bootstrap replicates for {metric}")
    return _interval(samples)


def per_site_metrics(data, n_bootstrap=1000, seed=20260916, min_class_count=10):
    """Return discrimination and case-clustered intervals for each site and finding."""
    generator = np.random.default_rng(seed)
    rows = []
    groups = data.groupby(["organ_group", "finding", "site"], sort=True)
    for (organ, finding, site), group in groups:
        positives = int(group["label"].sum())
        negatives = int(len(group) - positives)
        row = {
            "organ_group": organ,
            "finding": finding,
            "site": site,
            "cases": int(group["case_id"].nunique()),
            "slides_scored": int(len(group)),
            "positive_slides": positives,
            "negative_slides": negatives,
            "prevalence": positives / len(group),
            "estimable": positives >= min_class_count and negatives >= min_class_count,
        }
        row["reason_not_estimable"] = (
            "" if row["estimable"] else f"fewer than {min_class_count} slides in one class"
        )
        for metric in METRICS:
            if row["estimable"]:
                value = metric_value(group["label"], group["site_held_out_score"], metric)
                low, high = _bootstrap_site(group, "site_held_out_score", metric,
                                            n_bootstrap, generator)
            else:
                value = low = high = np.nan
            row[metric] = value
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _stratified_cases(group):
    return [(_case_indices(site.reset_index(drop=True)), len(site))
            for _, site in group.groupby("site", sort=True)]


def _stratified_draw(groups, generator):
    """Return concatenated row positions after resampling cases within sites."""
    selected = []
    offset = 0
    for case_indices, size in groups:
        selected.append(_draw_cases(case_indices, generator) + offset)
        offset += size
    return np.concatenate(selected)


def pooled_comparison(data, n_bootstrap=1000, seed=20260916):
    """Return pooled metrics and paired, site-stratified case-bootstrap intervals."""
    generator = np.random.default_rng(seed)
    rows = []
    for (organ, finding), group in data.groupby(["organ_group", "finding"], sort=True):
        # Site blocks make the offset arithmetic in ``_stratified_draw`` explicit.
        work = pd.concat([site.reset_index(drop=True)
                          for _, site in group.groupby("site", sort=True)], ignore_index=True)
        cases_by_site = _stratified_cases(group)
        labels = work["label"].to_numpy()
        fivefold_scores = work["fivefold_score"].to_numpy()
        site_scores = work["site_held_out_score"].to_numpy()
        row = {
            "organ_group": organ,
            "finding": finding,
            "cases": int(group["case_id"].nunique()),
            "slides_scored": int(len(group)),
            "positive_slides": int(group["label"].sum()),
            "prevalence": float(group["label"].mean()),
        }
        for metric in METRICS:
            fivefold = metric_value(labels, fivefold_scores, metric)
            site_held_out = metric_value(labels, site_scores, metric)
            boot_fivefold, boot_site, boot_difference = [], [], []
            attempts = 0
            while len(boot_difference) < n_bootstrap and attempts < n_bootstrap * 20:
                attempts += 1
                selected = _stratified_draw(cases_by_site, generator)
                sampled_labels = labels[selected]
                cv_value = metric_value(sampled_labels, fivefold_scores[selected], metric)
                site_value = metric_value(sampled_labels, site_scores[selected], metric)
                if np.isfinite(cv_value) and np.isfinite(site_value):
                    boot_fivefold.append(cv_value)
                    boot_site.append(site_value)
                    boot_difference.append(cv_value - site_value)
            if len(boot_difference) < n_bootstrap:
                raise ValueError(
                    f"only {len(boot_difference)} defined paired replicates for {metric}")
            cv_low, cv_high = _interval(boot_fivefold)
            site_low, site_high = _interval(boot_site)
            diff_low, diff_high = _interval(boot_difference)
            prefix = "auroc" if metric == "auroc" else "average_precision"
            row.update({
                f"fivefold_{prefix}": fivefold,
                f"fivefold_{prefix}_ci_low": cv_low,
                f"fivefold_{prefix}_ci_high": cv_high,
                f"site_held_out_{prefix}": site_held_out,
                f"site_held_out_{prefix}_ci_low": site_low,
                f"site_held_out_{prefix}_ci_high": site_high,
                f"difference_{prefix}": fivefold - site_held_out,
                f"difference_{prefix}_ci_low": diff_low,
                f"difference_{prefix}_ci_high": diff_high,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def site_average_comparison(data, n_bootstrap=1000, seed=20260916,
                            min_class_count=10):
    """Return unweighted site-average metrics and paired bootstrap intervals.

    The primary transportability estimand is the arithmetic mean of the
    site-specific metrics.  Each held-out site therefore contributes one
    estimate, rather than contributing in proportion to its number of slides.
    A site with fewer than ``min_class_count`` positive or negative slides for
    a finding has no estimate and is left out of that finding's average;
    ``sites`` counts the sites averaged and ``sites_total`` all sites.
    Bootstrap replicates resample cases within every averaged site and
    recompute the site average, preserving the equal-site weighting.  This is
    deliberately separate from :func:`pooled_comparison`, whose concatenated
    slide-level estimate is retained as a sensitivity analysis.
    """
    generator = np.random.default_rng(seed)
    rows = []
    for (organ, finding), group in data.groupby(["organ_group", "finding"], sort=True):
        site_blocks = []
        for site, site_group in group.groupby("site", sort=True):
            positives = int(site_group["label"].sum())
            negatives = int(len(site_group) - positives)
            if positives < min_class_count or negatives < min_class_count:
                continue
            work = site_group.reset_index(drop=True)
            site_blocks.append({
                "site": site,
                "work": work,
                "cases": _case_indices(work),
            })

        labels_by_site = [block["work"]["label"].to_numpy() for block in site_blocks]
        fivefold_by_site = [block["work"]["fivefold_score"].to_numpy()
                            for block in site_blocks]
        site_scores_by_site = [block["work"]["site_held_out_score"].to_numpy()
                               for block in site_blocks]
        n_sites = len(site_blocks)
        row = {
            "organ_group": organ,
            "finding": finding,
            "sites": n_sites,
            "sites_total": int(group["site"].nunique()),
            "cases": int(group["case_id"].nunique()),
            "slides_scored": int(len(group)),
            "positive_slides": int(group["label"].sum()),
            "prevalence": float(group["label"].mean()),
        }
        if n_sites == 0:
            rows.append(row)
            continue
        for metric in METRICS:
            site_fivefold = [metric_value(labels, scores, metric)
                             for labels, scores in zip(labels_by_site, fivefold_by_site)]
            site_held_out = [metric_value(labels, scores, metric)
                             for labels, scores in zip(labels_by_site, site_scores_by_site)]
            fivefold = float(np.mean(site_fivefold))
            site_held_out = float(np.mean(site_held_out))
            boot_fivefold, boot_site, boot_difference = [], [], []
            attempts = 0
            while len(boot_difference) < n_bootstrap and attempts < n_bootstrap * 20:
                attempts += 1
                replicate_fivefold = []
                replicate_site = []
                for block, labels, fivefold_scores, site_scores in zip(
                        site_blocks, labels_by_site, fivefold_by_site, site_scores_by_site):
                    selected = _draw_cases(block["cases"], generator)
                    sampled_labels = labels[selected]
                    cv_value = metric_value(sampled_labels, fivefold_scores[selected], metric)
                    site_value = metric_value(sampled_labels, site_scores[selected], metric)
                    if not np.isfinite(cv_value) or not np.isfinite(site_value):
                        break
                    replicate_fivefold.append(cv_value)
                    replicate_site.append(site_value)
                if len(replicate_fivefold) == n_sites:
                    cv_value = float(np.mean(replicate_fivefold))
                    site_value = float(np.mean(replicate_site))
                    boot_fivefold.append(cv_value)
                    boot_site.append(site_value)
                    boot_difference.append(cv_value - site_value)
            if len(boot_difference) < n_bootstrap:
                raise ValueError(
                    f"only {len(boot_difference)} defined site-average bootstrap "
                    f"replicates for {organ}/{finding}/{metric}"
                )
            cv_low, cv_high = _interval(boot_fivefold)
            site_low, site_high = _interval(boot_site)
            diff_low, diff_high = _interval(boot_difference)
            prefix = "auroc" if metric == "auroc" else "average_precision"
            row.update({
                f"fivefold_{prefix}": fivefold,
                f"fivefold_{prefix}_ci_low": cv_low,
                f"fivefold_{prefix}_ci_high": cv_high,
                f"site_held_out_{prefix}": site_held_out,
                f"site_held_out_{prefix}_ci_low": site_low,
                f"site_held_out_{prefix}_ci_high": site_high,
                f"difference_{prefix}": fivefold - site_held_out,
                f"difference_{prefix}_ci_low": diff_low,
                f"difference_{prefix}_ci_high": diff_high,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test_predictions(predictions, cohort_slides, folds_dir):
    """Pair the canonical headline and site-held-out test runs for analysis."""
    from champs_pipeline.eval.predictions import select_test_runs, check_test_folds

    fivefold = select_test_runs(predictions, "headline", "fivefold")
    site = select_test_runs(predictions, "site_held_out", "loso_nested")
    check_test_folds(fivefold, folds_dir)
    check_test_folds(site, folds_dir)
    keys = ["organ_group", "finding", "slide_id"]
    columns = keys + ["case_id", "y_true", "mask", "score"]
    paired = fivefold[columns].merge(site[columns], on=keys, how="outer",
                                     suffixes=("_cv", "_site"), indicator=True,
                                     validate="one_to_one")
    if not paired["_merge"].eq("both").all():
        raise ValueError("designs do not cover the same test cells")
    for name in ("case_id", "y_true", "mask"):
        left, right = paired[f"{name}_cv"], paired[f"{name}_site"]
        if not (left.eq(right) | (left.isna() & right.isna())).all():
            raise ValueError(f"designs disagree on {name}")
    paired = paired.loc[paired["mask_cv"].eq(1)]
    data = paired[keys].copy()
    data["case_id"] = paired["case_id_cv"]
    data["label"] = paired["y_true_cv"]
    data["fivefold_score"] = paired["score_cv"]
    data["site_held_out_score"] = paired["score_site"]
    sites = cohort_slides[["slide_id", "site"]]
    if sites.duplicated("slide_id").any():
        raise ValueError("cohort has duplicate slide ids")
    data = data.merge(sites, on="slide_id", how="left", validate="many_to_one")
    return validate_paired_predictions(data)
