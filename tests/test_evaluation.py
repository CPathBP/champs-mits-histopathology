"""Unit tests for small, stable components of the discrimination evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from champs_pipeline.eval import inference
from champs_pipeline.eval.discrimination import (grade_groups, macro_fold_metrics,
                                                metric_value, model_baseline_difference,
                                                non_image_baseline, per_grade_metrics,
                                                scored_slides, summarise_metrics,
                                                summary_over_folds)
from champs_pipeline.eval.display_data import display_inputs
from champs_pipeline.eval.predictions import check_test_folds, select_test_runs
from champs_pipeline.eval.transportability import (paired_test_predictions, per_site_metrics,
                                                    pooled_comparison, site_average_comparison,
                                                    validate_paired_predictions)
from champs_pipeline.figures.output import write_provenance
# The toy run of the inference tests; pytest finds the fixture through this import.
from test_inference import prepare, run_inputs  # noqa: F401


def test_summary_over_folds_defaults_to_sample_standard_deviation():
    """The reporting summary uses one value from each outer fold."""
    summary = summary_over_folds([0.6, 0.7, 0.8, 0.9, 1.0])
    assert np.isclose(summary["mean"], 0.8)
    assert np.isclose(summary["sd"], np.sqrt(0.025))
    assert summary["n_folds"] == 5


def test_summary_over_folds_can_use_student_t_interval():
    """The optional t summary remains available for interval analyses."""
    summary = summary_over_folds([0.6, 0.7, 0.8, 0.9, 1.0], method="t")
    assert np.isclose(summary["mean"], 0.8)
    assert summary["n_folds"] == 5
    assert np.isclose(summary["ci_low"], 0.6037, atol=1e-4)
    assert np.isclose(summary["ci_high"], 0.9963, atol=1e-4)


def test_summarise_metrics_orders_columns_the_same_in_every_process():
    """AUROC columns come before AP columns, whatever the process's hash seed."""
    scored = pd.DataFrame({"organ_group": "liver", "finding": ["steatosis"] * 4
                           + ["hemozoin_pigment"] * 4, "fold": [0, 0, 1, 1] * 2,
                           "label": [0, 1] * 4, "score": [0.2, 0.8] * 4})
    summary, _, _ = summarise_metrics(scored)
    columns = [column for column in summary.columns if column.endswith("_mean")]
    assert columns == ["auroc_mean", "average_precision_mean"]


def test_summary_over_folds_refuses_other_methods():
    with pytest.raises(ValueError, match="'sd' or 't'"):
        summary_over_folds([0.6, 0.7], method="bootstrap")


def test_metric_value_weights_count_repeated_rows():
    """A case weight of two equals the rows drawn twice; weight zero leaves a row out."""
    labels, scores = np.array([0, 1, 1, 0]), np.array([0.2, 0.9, 0.4, 0.5])
    weighted = metric_value(labels, scores, "auroc", weights=[2, 1, 1, 0])
    repeated = metric_value([0, 0, 1, 1], [0.2, 0.2, 0.9, 0.4], "auroc")
    assert np.isclose(weighted, repeated)
    assert np.isnan(metric_value([1, 1], [0.2, 0.9], "auroc"))


def test_scored_slides_refuses_rows_that_are_not_test_rows():
    predictions = pd.DataFrame({"split": ["val"], "slide_id": ["a"]})
    with pytest.raises(ValueError, match="test predictions"):
        scored_slides(predictions, pd.DataFrame(), "elig")


def test_macro_fold_metrics_requires_every_study_finding():
    scored = pd.DataFrame({"organ_group": "liver", "finding": "steatosis", "fold": 0,
                           "label": [0, 1], "score": [0.1, 0.9]})
    with pytest.raises(ValueError, match="expected metrics"):
        macro_fold_metrics(scored)


def _baseline_inputs(tmp_path):
    """Toy cohort in which the site alone separates positive from negative slides."""
    slides = [f"s{index}" for index in range(12)]
    sites = ["Kenya", "Mali"] * 6
    labels = [int(site == "Kenya") for site in sites]
    splits = ["train"] * 8 + ["test"] * 4
    fold = tmp_path / "lung_elig_fivefold" / "fold_0.csv"
    fold.parent.mkdir()
    pd.DataFrame({"slide_id": slides, "split": splits}).to_csv(fold, index=False)
    reference = pd.DataFrame({"organ_group": "lung", "slide_id": slides,
                              "finding": "bronchopneumonia", "variant": "elig",
                              "label": labels, "severity": np.nan})
    cohort_slides = pd.DataFrame({"slide_id": slides, "champs_deid": slides, "site": sites,
                                  "slide_source": "CPL", "scanner_power": 20,
                                  "n_patches_virchow2": 1000})
    cohort_cases = pd.DataFrame({"champs_deid": slides, "death_category": "neonate",
                                 "age_months_total": 1.0, "calc_postmortem_hrs": np.nan,
                                 "death_year": 2020})
    test = [index for index, split in enumerate(splits) if split == "test"]
    scored = pd.DataFrame({"organ_group": "lung", "finding": "bronchopneumonia", "fold": 0,
                           "slide_id": [slides[index] for index in test],
                           "label": [labels[index] for index in test], "score": 0.5})
    return scored, reference, cohort_cases, cohort_slides


def test_non_image_baseline_fits_on_training_slides_and_scores_test_slides(tmp_path):
    scored, reference, cases, slides = _baseline_inputs(tmp_path)
    baseline = non_image_baseline(scored, reference, cases, slides, tmp_path, "elig")
    assert baseline["slide_id"].tolist() == scored["slide_id"].tolist()
    assert metric_value(scored["label"], baseline["baseline_score"], "auroc") == 1.0


def test_non_image_baseline_refuses_slides_outside_the_test_split(tmp_path):
    scored, reference, cases, slides = _baseline_inputs(tmp_path)
    scored.loc[0, "slide_id"] = "s0"
    with pytest.raises(ValueError, match="not held-out test slides"):
        non_image_baseline(scored, reference, cases, slides, tmp_path, "elig")


def test_grade_groups_combine_severe_and_extensive_and_keep_ungraded_positives():
    """Grade groups follow the prespecified reporting categories."""
    frame = pd.DataFrame({"label": [0, 1, 1, 1, 1, 1],
                          "severity": [np.nan, "minimal", "mild", "moderate", "extensive", np.nan]})
    assert grade_groups(frame)["grade_group"].astype(str).tolist() == [
        "report-negative", "minimal", "mild", "moderate", "severe or extensive", "not graded"]


def test_scored_slides_joins_labels_and_drops_masked_rows():
    """A masked Reference A record cannot contribute to a discrimination metric."""
    predictions = pd.DataFrame({"run_id": ["r", "r"], "organ_group": ["lung", "lung"],
                                "label_variant": ["elig", "elig"],
                                "design": ["fivefold", "fivefold"],
                                "fold": [0, 0], "slide_id": ["a", "b"],
                                "finding": ["bronchopneumonia", "bronchopneumonia"],
                                "score": [0.1, 0.9], "split": ["test", "test"]})
    reference = pd.DataFrame({"organ_group": ["lung", "lung"], "slide_id": ["a", "b"],
                              "finding": ["bronchopneumonia", "bronchopneumonia"],
                              "variant": ["elig", "elig"], "label": [0, np.nan],
                              "severity": [np.nan, np.nan]})
    result = scored_slides(predictions, reference, "elig")
    assert result[["slide_id", "label"]].to_dict("records") == [{"slide_id": "a", "label": 0}]


def test_per_grade_metrics_reports_only_folds_that_contributed():
    """A grade absent from one fold has an explicitly smaller fold count."""
    scored = pd.DataFrame({
        "organ_group": ["lung", "lung", "lung", "lung", "lung"],
        "finding": ["bronchopneumonia"] * 5,
        "fold": [0, 0, 1, 1, 1],
        "label": [0, 1, 0, 0, 1],
        "severity": [np.nan, "minimal", np.nan, np.nan, "mild"],
        "score": [0.1, 0.9, 0.1, 0.2, 0.8],
    })
    summary, fold = per_grade_metrics(scored)
    summary = summary.set_index("grade_group")
    assert summary.loc["minimal", "n_folds"] == 1
    assert summary.loc["mild", "n_folds"] == 1
    assert fold.set_index("grade_group").loc["minimal", "fold"] == 0


def test_model_baseline_difference_adds_paired_macro_average():
    rows = []
    baseline_rows = []
    for finding, model_scores, baseline_scores in [
        ("a", [0.1, 0.9], [0.2, 0.8]),
        ("b", [0.2, 0.8], [0.9, 0.1]),
    ]:
        for fold in range(2):
            for index, label in enumerate([0, 1]):
                common = {"organ_group": "lung", "finding": finding, "fold": fold,
                          "slide_id": f"{finding}-{fold}-{index}"}
                rows.append({**common, "label": label, "score": model_scores[index]})
                baseline_rows.append({**common, "baseline_score": baseline_scores[index]})
    summary, _ = model_baseline_difference(pd.DataFrame(rows), pd.DataFrame(baseline_rows))
    macro = summary.set_index(["organ_group", "finding"]).loc[("lung", "macro")]
    assert np.isclose(macro["difference_mean"], 0.5)


def _paired_transportability_rows():
    rows = []
    for site in ("BD", "ET"):
        for case in range(20):
            label = case >= 10
            rows.append({"organ_group": "lung", "finding": "bronchopneumonia",
                         "slide_id": f"{site}-{case}", "case_id": f"{site}-case-{case}",
                         "site": site, "label": int(label),
                         "fivefold_score": 0.9 if label else 0.1,
                         "site_held_out_score": 0.8 if label else 0.2})
    return pd.DataFrame(rows)


def test_transportability_reports_sites_and_paired_design_difference():
    paired = validate_paired_predictions(_paired_transportability_rows())
    site = per_site_metrics(paired, n_bootstrap=20, seed=1)
    comparison = pooled_comparison(paired, n_bootstrap=20, seed=1)
    assert set(site["site"]) == {"BD", "ET"}
    assert site["estimable"].all()
    assert np.allclose(site["auroc"], 1)
    assert comparison.loc[0, "fivefold_auroc"] == 1
    assert comparison.loc[0, "site_held_out_auroc"] == 1
    assert comparison.loc[0, "difference_auroc"] == 0


def test_transportability_site_average_gives_each_site_equal_weight():
    paired = validate_paired_predictions(_paired_transportability_rows())
    comparison = site_average_comparison(paired, n_bootstrap=20, seed=1)
    assert comparison.loc[0, "sites"] == 2
    assert comparison.loc[0, "site_held_out_auroc"] == 1
    assert comparison.loc[0, "fivefold_auroc"] == 1
    assert comparison.loc[0, "difference_auroc"] == 0


def test_transportability_marks_sparse_site_not_estimable():
    paired = validate_paired_predictions(_paired_transportability_rows().iloc[:15])
    site = per_site_metrics(paired, n_bootstrap=5, seed=1)
    assert not bool(site.loc[0, "estimable"])
    assert np.isnan(site.loc[0, "auroc"])


def test_transportability_site_average_leaves_out_sparse_sites():
    rows = _paired_transportability_rows()
    sparse = rows[(rows["site"] == "BD") | (rows["label"] == 0) | (rows["slide_id"] == "ET-19")]
    comparison = site_average_comparison(validate_paired_predictions(sparse), n_bootstrap=20,
                                         seed=1)
    assert comparison.loc[0, "sites"] == 1
    assert comparison.loc[0, "sites_total"] == 2
    assert comparison.loc[0, "site_held_out_auroc"] == 1


def test_transportability_rejects_a_case_split_across_sites():
    paired = _paired_transportability_rows()
    paired.loc[20, "case_id"] = paired.loc[0, "case_id"]
    with pytest.raises(ValueError, match="case is assigned to more than one site"):
        validate_paired_predictions(paired)


def test_consumers_select_test_family_and_pair_designs(run_inputs):
    frame = inference.predict_run(prepare(run_inputs))
    test = select_test_runs(frame, "headline", "fivefold")
    assert len(test) == 4
    check_test_folds(test, run_inputs["root"] / "folds")
    site = frame.assign(run_id="toy-site", families="site_held_out", design="loso_nested")
    site_dir = run_inputs["root"] / "folds/lung_elig_loso_nested"
    site_dir.mkdir()
    (site_dir / "fold_0.csv").write_bytes(run_inputs["fold"].read_bytes())
    cohort = pd.DataFrame({"slide_id": frame["slide_id"].unique(), "site": "toy-site"})
    paired = paired_test_predictions(pd.concat([frame, site]), cohort, run_inputs["root"] / "folds")
    assert len(paired) == 3
    assert paired["fivefold_score"].equals(paired["site_held_out_score"])
    ambiguous = pd.concat([frame, frame.assign(run_id="another-model")])
    with pytest.raises(ValueError, match="multiple models"):
        select_test_runs(ambiguous, "headline", "fivefold")


def test_draft_displays_record_draft_and_omit_the_record_input(run_inputs, tmp_path):
    predictions = run_inputs["root"] / "draft.parquet"
    reference = run_inputs["root"] / "reference.parquet"
    for path in (predictions, reference):
        path.write_bytes(b"toy")
    final = display_inputs(predictions, reference)
    draft = display_inputs(predictions, reference, draft=True)
    assert predictions.with_suffix(".json") in final
    assert draft == [predictions, reference]
    record = write_provenance(tmp_path, "toy-display", draft, draft=True)
    assert record["draft"] is True
    assert json.loads((tmp_path / "toy-display.provenance.json").read_text())["draft"] is True
    assert write_provenance(tmp_path, "toy-final", draft)["draft"] is False

