"""Cross-validated discrimination metrics, grade groups and non-image baseline.

All summaries preserve the outer-fold structure.  This prevents a slide-rich
fold from receiving more weight than another fold and makes the reported
summary a description of variation across held-out folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from champs_pipeline.figures.labels import FINDINGS


GRADE_ORDER = ["report-negative", "minimal", "mild", "moderate", "severe or extensive",
               "not graded"]
GRADE_LABELS = {
    "report-negative": "Report-negative", "minimal": "Minimal", "mild": "Mild",
    "moderate": "Moderate", "severe or extensive": "Severe or extensive",
    "not graded": "Not graded",
}
METRICS = {"auroc", "average_precision"}
NUMERIC_FEATURES = ["age_months_total", "calc_postmortem_hrs", "death_year",
                    "n_patches_virchow2"]
CATEGORICAL_FEATURES = ["site", "slide_source", "scanner_power", "death_category"]
FOLD_KEYS = ["organ_group", "finding", "fold"]


def scored_slides(predictions, reference_slides, evaluation_variant):
    """Join predictions to one Reference A label variant and drop masked slides.

    The returned rows carry the held-out fold, score, binary label and recorded
    severity.  Predictions without a matching reference record are omitted;
    this is intentional because they cannot contribute to a discrimination
    estimate.  A duplicate reference key is an input error.
    """
    if "split" not in predictions or not predictions["split"].eq("test").all():
        raise ValueError("discrimination requires explicitly selected test predictions")
    columns = ["organ_group", "slide_id", "finding", "label", "severity"]
    reference = reference_slides.loc[reference_slides["variant"] == evaluation_variant,
                                     columns].copy()
    keys = ["organ_group", "slide_id", "finding"]
    if reference.duplicated(keys).any():
        raise ValueError("Reference A has duplicate organ, slide and finding records")
    joined = predictions.merge(reference, on=keys, how="inner", validate="many_to_one")
    joined = joined.loc[joined["label"].notna()].copy()
    joined["label"] = joined["label"].astype(int)
    if set(joined["label"]) - {0, 1}:
        raise ValueError("Reference A labels must be 0, 1 or masked")
    if joined.duplicated(["run_id", "slide_id", "finding"]).any():
        raise ValueError("the scored data contain duplicate prediction rows")
    return joined


def metric_value(labels, scores, metric, weights=None):
    """Compute one discrimination metric, returning NaN when one class is absent.

    ``weights`` are case-bootstrap multiplicities; rows of weight zero are left out.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        kept = weights > 0
        labels, scores, weights = labels[kept], scores[kept], weights[kept]
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        return np.nan
    if metric == "auroc":
        return float(roc_auc_score(labels, scores, sample_weight=weights))
    return float(average_precision_score(labels, scores, sample_weight=weights))


def per_fold_metrics(scored, score_column="score"):
    """Return slide counts, prevalence and discrimination metrics by outer fold."""
    rows = []
    for (organ, finding, fold), group in scored.groupby(FOLD_KEYS, sort=True):
        rows.append({"organ_group": organ, "finding": finding, "fold": int(fold),
                     "slides_scored": len(group), "positive_slides": int(group["label"].sum()),
                     "auroc": metric_value(group["label"], group[score_column], "auroc"),
                     "average_precision": metric_value(group["label"], group[score_column],
                                                        "average_precision")})
    return pd.DataFrame(rows)


def macro_fold_metrics(scored, score_column="score"):
    """Return fold metrics averaged equally over the study findings of each organ."""
    finding_metrics = per_fold_metrics(scored, score_column)
    rows = []
    for (organ, fold), group in finding_metrics.groupby(["organ_group", "fold"], sort=True):
        expected = set(FINDINGS[organ])
        if set(group["finding"]) != expected:
            raise ValueError(f"{organ} fold {fold}: expected metrics for {sorted(expected)}")
        rows.append({"organ_group": organ, "finding": "macro", "fold": int(fold),
                     "slides_scored": np.nan, "positive_slides": np.nan,
                     "auroc": group["auroc"].mean(),
                     "average_precision": group["average_precision"].mean()})
    return pd.DataFrame(rows)


def summary_over_folds(values, method="sd"):
    """Summarise one metric over the outer folds.

    ``method='sd'`` (the reporting default) gives the mean and the sample
    standard deviation (``ddof=1``) of the fold estimates; ``method='t'`` gives
    the mean with a 95% Student t interval on k - 1 degrees of freedom.
    """
    if method not in {"sd", "t"}:
        raise ValueError("summary method must be 'sd' or 't'")
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) < 2:
        mean = float(values[0]) if len(values) else np.nan
        spread = {"sd": np.nan} if method == "sd" else {"ci_low": np.nan, "ci_high": np.nan}
        return {"mean": mean, **spread, "n_folds": len(values)}
    mean = float(values.mean())
    if method == "sd":
        return {"mean": mean, "sd": float(values.std(ddof=1)), "n_folds": len(values)}
    half_width = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1)
                       / np.sqrt(len(values)))
    return {"mean": mean, "ci_low": mean - half_width, "ci_high": mean + half_width,
            "n_folds": len(values)}


def summarise_metrics(scored, score_column="score", summary="sd"):
    """Summarise AUROC and AP for every finding and each organ macro average."""
    fold_metrics = per_fold_metrics(scored, score_column)
    rows = []
    for (organ, finding), group in fold_metrics.groupby(["organ_group", "finding"], sort=True):
        row = {"organ_group": organ, "finding": finding,
               "slides_scored": int(group["slides_scored"].sum()),
               "positive_slides": int(group["positive_slides"].sum())}
        for metric in METRICS:
            metric_summary = summary_over_folds(group[metric], method=summary)
            row.update({f"{metric}_{name}": value for name, value in metric_summary.items()})
        rows.append(row)
    macro = macro_fold_metrics(scored, score_column)
    for organ, group in macro.groupby("organ_group", sort=True):
        row = {"organ_group": organ, "finding": "macro", "slides_scored": np.nan,
               "positive_slides": np.nan}
        for metric in METRICS:
            metric_summary = summary_over_folds(group[metric], method=summary)
            row.update({f"{metric}_{name}": value for name, value in metric_summary.items()})
        rows.append(row)
    return pd.DataFrame(rows), fold_metrics, macro


def grade_groups(scored):
    """Add the prespecified severity group to scored Reference A rows."""
    result = scored.copy()
    result["grade_group"] = np.where(result["label"] == 0, "report-negative", "not graded")
    positive = result["label"] == 1
    severity = result.loc[positive, "severity"]
    valid = {"minimal", "mild", "moderate", "severe", "extensive"}
    unexpected = set(severity.dropna()) - valid
    if unexpected:
        raise ValueError(f"unrecognised severity values: {sorted(unexpected)}")
    result.loc[positive & result["severity"].isin(["minimal", "mild", "moderate"]),
               "grade_group"] = result.loc[positive & result["severity"].isin(
                   ["minimal", "mild", "moderate"]), "severity"]
    result.loc[positive & result["severity"].isin(["severe", "extensive"]),
               "grade_group"] = "severe or extensive"
    result["grade_group"] = pd.Categorical(result["grade_group"], GRADE_ORDER, ordered=True)
    return result


def per_grade_metrics(scored, score_column="score", summary="sd"):
    """Compare each positive grade group with report-negative slides in every fold.

    The returned summary includes ``n_folds`` from ``summary_over_folds``. A
    grade absent from a test fold has no fold-level comparison, so its count can
    be smaller than the study's five outer folds and contextualises its interval.
    """
    grouped = grade_groups(scored)
    rows = []
    positive_rows = grouped[grouped["grade_group"] != "report-negative"]
    for (organ, finding, grade, fold), positives in positive_rows.groupby(
            ["organ_group", "finding", "grade_group", "fold"], observed=True):
        negatives = grouped[(grouped["organ_group"] == organ) & (grouped["finding"] == finding)
                            & (grouped["fold"] == fold)
                            & (grouped["grade_group"] == "report-negative")]
        comparison = pd.concat([negatives, positives])
        rows.append({"organ_group": organ, "finding": finding, "grade_group": str(grade),
                     "fold": int(fold), "grade_slides": len(positives),
                     "negative_slides": len(negatives),
                     "auroc": metric_value(comparison["label"], comparison[score_column], "auroc")})
    fold = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in fold.groupby(["organ_group", "finding", "grade_group"], sort=True):
        grade_summary = summary_over_folds(group["auroc"], method=summary)
        summary_rows.append({"organ_group": keys[0], "finding": keys[1], "grade_group": keys[2],
                             "grade_slides": int(group["grade_slides"].sum()),
                             "negative_slides": int(group["negative_slides"].sum()),
                             **grade_summary})
    return pd.DataFrame(summary_rows), fold


def _fold_path(folds_dir, organ, variant, fold):
    path = Path(folds_dir) / f"{organ}_{variant}_fivefold" / f"fold_{fold}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing fold definition: {path}")
    return path


def non_image_baseline(scored, reference_slides, cohort_cases, cohort_slides, folds_dir,
                       evaluation_variant):
    """Fit and score the prespecified non-image logistic-regression baseline.

    Every fold and finding is fitted on that fold's unmasked Reference A train
    slides.  Categorical fields are one-hot encoded; numeric fields receive a
    median imputation and missingness indicator, then are standardised using
    the training slides only.  Tissue amount is log-transformed as specified.
    """
    slide_columns = ["slide_id", "champs_deid", "site", "slide_source", "scanner_power",
                     "n_patches_virchow2"]
    features = cohort_slides.loc[:, slide_columns].merge(
        cohort_cases.loc[:, ["champs_deid", "death_category", "age_months_total",
                             "calc_postmortem_hrs", "death_year"]],
        on="champs_deid", how="left", validate="many_to_one")
    features["n_patches_virchow2"] = np.log(features["n_patches_virchow2"].clip(lower=1))
    reference = reference_slides[reference_slides["variant"] == evaluation_variant].copy()
    reference = reference[reference["label"].notna()]
    records = []
    for (organ, finding, fold), test in scored.groupby(FOLD_KEYS, sort=True):
        split_path = _fold_path(folds_dir, organ, evaluation_variant, fold)
        splits = pd.read_csv(split_path, usecols=["slide_id", "split"])
        if splits.duplicated("slide_id").any() or set(splits["split"]) - {"train", "val", "test"}:
            raise ValueError(f"invalid split file: {split_path}")
        expected_test = set(splits.loc[splits["split"] == "test", "slide_id"].astype(str))
        observed_test = set(test["slide_id"].astype(str))
        if not observed_test.issubset(expected_test):
            raise ValueError(f"{organ} fold {fold}: predictions are not held-out test slides")
        labels = reference[(reference["organ_group"] == organ) & (reference["finding"] == finding)]
        labelled = labels.merge(splits, on="slide_id", how="inner", validate="one_to_one")
        train = labelled[labelled["split"] == "train"].merge(features, on="slide_id", how="left",
                                                                validate="one_to_one")
        test_features = test.merge(features, on="slide_id", how="left", validate="one_to_one")
        transformer = ColumnTransformer([
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                        ("one_hot", OneHotEncoder(handle_unknown="ignore"))]),
             CATEGORICAL_FEATURES),
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                    ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ])
        if train["label"].nunique() == 1:
            scores = np.repeat(float(train["label"].iloc[0]), len(test_features))
        else:
            model = Pipeline([("features", transformer),
                              ("model", LogisticRegression(penalty="l2", max_iter=10000))])
            model.fit(train[CATEGORICAL_FEATURES + NUMERIC_FEATURES], train["label"])
            predictors = test_features[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
            scores = model.predict_proba(predictors)[:, 1]
        records.extend({"organ_group": organ, "finding": finding, "fold": int(fold),
                        "slide_id": slide_id, "baseline_score": float(score)}
                       for slide_id, score in zip(test["slide_id"], scores))
    return pd.DataFrame(records)


def model_baseline_difference(scored, baseline, summary="sd"):
    """Summarise paired per-fold AUROC differences, model minus context baseline."""
    joined = scored.merge(baseline, on=["organ_group", "finding", "fold", "slide_id"],
                          how="inner", validate="one_to_one")
    rows = []
    for (organ, finding, fold), group in joined.groupby(FOLD_KEYS, sort=True):
        rows.append({"organ_group": organ, "finding": finding, "fold": int(fold),
                     "difference": metric_value(group["label"], group["score"], "auroc")
                     - metric_value(group["label"], group["baseline_score"], "auroc")})
    fold = pd.DataFrame(rows)
    summary_rows = []
    for (organ, finding), group in fold.groupby(["organ_group", "finding"], sort=True):
        values = summary_over_folds(group["difference"], method=summary)
        summary_rows.append({"organ_group": organ, "finding": finding,
                             **{f"difference_{name}": value
                                for name, value in values.items()}})
    macro = (fold.groupby(["organ_group", "fold"], as_index=False)["difference"]
             .mean())
    for organ, group in macro.groupby("organ_group", sort=True):
        values = summary_over_folds(group["difference"], method=summary)
        summary_rows.append({"organ_group": organ, "finding": "macro",
                             **{f"difference_{name}": value
                                for name, value in values.items()}})
    return pd.DataFrame(summary_rows), fold

