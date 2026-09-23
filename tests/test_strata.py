"""Tests for discrimination within strata and the covariate-adjusted AUROC, on toy data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from champs_pipeline.eval.strata import (NOT_RECORDED, adjusted_auroc, attach_strata,
                                         slide_strata, stratum_metrics)


def scored_frame(n_slides=600, seed=0, shift=0.0):
    """Toy scored predictions for one finding with two levels of one stratum.

    ``shift`` adds a level-wide score offset so that the pooled AUROC exceeds
    the within-level AUROC when the level also carries a prevalence difference.
    """
    generator = np.random.default_rng(seed)
    level = np.where(np.arange(n_slides) < n_slides // 2, "A", "B")
    prevalence = np.where(level == "A", 0.1, 0.7)
    labels = (generator.random(n_slides) < prevalence).astype(int)
    noise = generator.normal(0.3, 0.2, n_slides)
    scores = np.clip(labels * 0.3 + noise + shift * (level == "B"), 0, 1)
    return pd.DataFrame({
        "organ_group": "lung",
        "finding": "pneumonitis",
        "fold": np.arange(n_slides) % 5,
        "slide_id": [f"s{i}" for i in range(n_slides)],
        "case_id": [f"c{i // 2}" for i in range(n_slides)],
        "label": labels,
        "score": scores,
        "level": level,
    })


def cohort_tables():
    """Toy cohort tables: five training slides of four cases and one other slide."""
    slides = pd.DataFrame({
        "slide_id": [f"s{i}" for i in range(6)],
        "champs_deid": ["c0", "c0", "c1", "c1", "c2", "c3"],
        "organ_group": "lung",
        "site": ["Kenya", "Kenya", "Sierra_Leone", "Mali", "Mali", "Nigeria_Cross_River"],
        "slide_source": ["CPL", "SITE", "SITE", "CPL", "SITE", "SITE"],
        "n_patches_virchow2": [100, 200, 300, 400, 500, 600],
        "training": [True, True, True, True, True, False],
    })
    cases = pd.DataFrame({
        "champs_deid": ["c0", "c1", "c2", "c3"],
        "age_group": ["Stillbirth", "Child (12 months to less than 60 Months)", np.nan,
                      "Stillbirth"],
        "calc_postmortem_hrs": [3.0, 30.0, np.nan, 5.0],
        "death_year": [2018, 2023, 2021, 2024],
    })
    return slides, cases


def toy_scored(slide_ids, case_ids):
    """Toy scored rows for the given slides and their cases."""
    return pd.DataFrame({
        "slide_id": slide_ids,
        "case_id": case_ids,
        "organ_group": "lung",
        "finding": "pneumonitis",
        "label": 0,
        "score": 0.5,
    })


def test_adjusted_auroc_is_positive_weighted_mean_of_within_level_values():
    scored = scored_frame(shift=0.3)
    result = adjusted_auroc(scored, "level", n_bootstrap=50, seed=1).iloc[0]
    within = {}
    positives = {}
    for level, group in scored.groupby("level"):
        within[level] = roc_auc_score(group["label"], group["score"])
        positives[level] = group["label"].sum()
    expected = sum(within[level] * positives[level] for level in within) / sum(positives.values())
    assert result["adjusted_auroc"] == pytest.approx(expected)
    assert result["pooled_auroc"] == pytest.approx(roc_auc_score(scored["label"], scored["score"]))
    assert result["pooled_auroc"] > result["adjusted_auroc"]
    assert result["difference_auroc_ci_low"] <= result["difference_auroc"]
    assert result["difference_auroc"] <= result["difference_auroc_ci_high"]
    assert result["prevalence_auroc"] > 0.5


def test_adjusted_equals_pooled_with_one_level():
    scored = scored_frame()
    scored["level"] = "A"
    result = adjusted_auroc(scored, "level", n_bootstrap=20, seed=2).iloc[0]
    assert result["adjusted_auroc"] == pytest.approx(result["pooled_auroc"])
    assert result["prevalence_auroc"] == pytest.approx(0.5)
    assert result["levels_used"] == 1


def test_levels_without_both_classes_are_excluded():
    scored = scored_frame()
    scored.loc[scored["level"] == "A", "label"] = 0
    result = adjusted_auroc(scored, "level", n_bootstrap=20, seed=3).iloc[0]
    assert result["levels_used"] == 1
    assert result["levels_total"] == 2
    assert result["slides_excluded"] == (scored["level"] == "A").sum()


def test_not_recorded_level_is_excluded_from_the_adjusted_auroc():
    scored = scored_frame()
    scored.loc[scored["level"] == "A", "level"] = NOT_RECORDED
    result = adjusted_auroc(scored, "level", n_bootstrap=20, seed=3).iloc[0]
    assert result["levels_used"] == 1
    assert result["slides_excluded"] == (scored["level"] == NOT_RECORDED).sum()


def test_stratum_metrics_marks_sparse_levels():
    scored = scored_frame()
    scored.loc[scored["level"] == "A", "label"] = 0
    scored.loc[scored["slide_id"].isin(["s0", "s1", "s2"]), "label"] = 1
    metrics = stratum_metrics(scored, "level", n_bootstrap=30, seed=4).set_index("level")
    assert not metrics.loc["A", "estimable"]
    assert metrics.loc["A", "positive_slides"] == 3
    assert np.isnan(metrics.loc["A", "auroc"])
    level_b = metrics.loc["B"]
    assert level_b["estimable"]
    assert level_b["auroc_ci_low"] <= level_b["auroc"] <= level_b["auroc_ci_high"]
    assert level_b["positive_slides"] == scored.loc[scored["level"] == "B", "label"].sum()


def test_stratum_metrics_repeat_with_the_same_seed():
    scored = scored_frame()
    first = stratum_metrics(scored, "level", n_bootstrap=30, seed=5)
    second = stratum_metrics(scored, "level", n_bootstrap=30, seed=5)
    pd.testing.assert_frame_equal(first, second)


def test_slide_strata_levels():
    slides, cases = cohort_tables()
    strata, cut_points = slide_strata(slides, cases)
    strata = strata.set_index("slide_id")
    assert list(strata.index) == ["s0", "s1", "s2", "s3", "s4"]
    assert strata.loc["s4", "age_group"] == NOT_RECORDED
    assert strata.loc["s4", "postmortem_interval"] == NOT_RECORDED
    assert list(strata["postmortem_interval"][:4]) == ["≤6 h", "≤6 h", ">24 h", ">24 h"]
    assert list(strata["year_of_death"]) == ["2016–2019", "2016–2019", "2022–2024",
                                             "2022–2024", "2020–2021"]
    assert list(strata["scan_source"][:2]) == ["Central laboratory", "Site"]
    assert set(strata["tissue"]) == {"Lowest tertile", "Middle tertile", "Highest tertile"}
    assert cut_points["lung"][0] == 100
    assert cut_points["lung"][-1] == 500


def test_slide_strata_maps_site_names_to_codes():
    slides, cases = cohort_tables()
    strata, _ = slide_strata(slides, cases)
    assert list(strata["site"]) == ["KE", "KE", "SL", "ML", "ML"]


def test_slide_strata_rejects_an_unknown_training_site():
    slides, cases = cohort_tables()
    slides.loc[5, "training"] = True
    with pytest.raises(ValueError, match="unknown site"):
        slide_strata(slides, cases)


def test_attach_strata_keeps_one_agreeing_case():
    strata, _ = slide_strata(*cohort_tables())
    joined = attach_strata(toy_scored(["s0", "s2"], ["c0", "c1"]), strata)
    assert list(joined["case_id"]) == ["c0", "c1"]
    assert not {"case_id_x", "case_id_y"} & set(joined.columns)
    assert list(joined["site"]) == ["KE", "SL"]


def test_attach_strata_rejects_another_case():
    strata, _ = slide_strata(*cohort_tables())
    with pytest.raises(ValueError, match="another case"):
        attach_strata(toy_scored(["s0", "s2"], ["c0", "c9"]), strata)


def test_attach_strata_rejects_a_slide_without_strata():
    strata, _ = slide_strata(*cohort_tables())
    with pytest.raises(ValueError, match="no strata"):
        attach_strata(toy_scored(["s0", "s5"], ["c0", "c3"]), strata)
