"""Tests for the modelling-choice comparisons and the learning-curve summary on toy data."""

import json

import numpy as np
import pandas as pd
import pytest

from champs_pipeline.eval.modelling_choices import (family_predictions, learning_curve_folds,
                                                    learning_curve_summary,
                                                    paired_differences, training_counts,
                                                    variant_summaries)


LUNG = ["aspiration_squames", "aspiration_meconium", "bronchopneumonia", "pneumonitis",
        "hyaline_membranes"]
FINDING = "bronchopneumonia"
FOLDS = 2
SLIDES_PER_FOLD = 4
TOY_RUNS = [
    {"families": "headline;aggregator_comparison;learning_curve", "encoder": "virchow2",
     "aggregator": "clam_mb", "train_fraction": 1.0},
    {"families": "aggregator_comparison", "encoder": "virchow2", "aggregator": "meanpool",
     "train_fraction": 1.0},
    {"families": "encoder_comparison", "encoder": "uni_v2", "aggregator": "clam_mb",
     "train_fraction": 1.0},
    {"families": "learning_curve", "encoder": "virchow2", "aggregator": "clam_mb",
     "train_fraction": 0.5},
]


def scored_frame(variants, n_slides=200, seed=0):
    """Toy scored predictions for several variants of the lung, with noisy scores."""
    generator = np.random.default_rng(seed)
    rows = []
    labels = {finding: generator.integers(0, 2, n_slides) for finding in LUNG}
    folds = np.arange(n_slides) % 5
    for variant, noise in variants.items():
        for finding in LUNG:
            noisy = labels[finding] * 0.6 + generator.normal(0.2, noise, n_slides)
            scores = np.clip(noisy, 0, 1)
            for slide in range(n_slides):
                rows.append({"run_id": f"{variant}_{folds[slide]}", "organ_group": "lung",
                             "label_variant": "elig", "design": "fivefold",
                             "fold": int(folds[slide]), "slide_id": f"s{slide}",
                             "case_id": f"c{slide // 2}", "finding": finding,
                             "label": int(labels[finding][slide]), "severity": "",
                             "score": float(scores[slide]), "variant": variant})
    return pd.DataFrame(rows)


def write_folds(root):
    """Write toy fold manifests of the lung and return their directory and frames."""
    folds_dir = root / "folds"
    directory = folds_dir / "lung_elig_fivefold"
    directory.mkdir(parents=True)
    slides = np.arange(FOLDS * SLIDES_PER_FOLD)
    manifests = {}
    for fold in range(FOLDS):
        manifest = pd.DataFrame({
            "slide_id": [f"s{slide}" for slide in slides],
            "case_id": [f"c{slide // 2}" for slide in slides],
            "split": np.where(slides // SLIDES_PER_FOLD == fold, "test", "train"),
            f"label_{FINDING}": (slides % 2).astype(float),
            f"mask_{FINDING}": 1,
        })
        manifest.to_csv(directory / f"fold_{fold}.csv", index=False)
        manifests[fold] = manifest
    return folds_dir, manifests


def run_rows(run, fold, manifest):
    """Toy test predictions of one run on one fold, consistent with its manifest."""
    test = manifest[manifest["split"] == "test"]
    run_id = f"{run['encoder']}-{run['aggregator']}-{run['train_fraction']}-fold{fold}"
    return pd.DataFrame({"run_id": run_id, **run, "organ_group": "lung",
                         "label_variant": "elig", "design": "fivefold", "fold": fold,
                         "learning_rate": 1e-4, "seed": 42, "slide_id": test["slide_id"],
                         "case_id": test["case_id"], "split": "test", "finding": FINDING,
                         "y_true": test[f"label_{FINDING}"], "mask": 1, "score": 0.5,
                         "logit": 0.0})


@pytest.fixture
def toy_table(tmp_path):
    """A toy prediction table of four runs on two folds and its fold manifests."""
    folds_dir, manifests = write_folds(tmp_path)
    frames = [run_rows(run, fold, manifests[fold]) for run in TOY_RUNS for fold in manifests]
    return pd.concat(frames, ignore_index=True), folds_dir


def test_family_selects_its_runs_and_names_the_variant(toy_table):
    table, folds_dir = toy_table
    aggregators = family_predictions(table, "aggregator_comparison", folds_dir)
    assert set(aggregators["variant"]) == {"clam_mb", "meanpool"}
    assert aggregators["split"].eq("test").all()
    curve = family_predictions(table, "learning_curve", folds_dir)
    assert set(curve["variant"]) == {0.5, 1.0}


def test_encoder_family_adds_the_headline_reference_run(toy_table):
    table, folds_dir = toy_table
    encoders = family_predictions(table, "encoder_comparison", folds_dir)
    assert set(encoders["variant"]) == {"virchow2", "uni_v2"}
    headline = encoders[encoders["variant"] == "virchow2"]
    assert headline["families"].str.contains("headline").all()


def test_family_refuses_two_models_of_one_variant(toy_table):
    table, folds_dir = toy_table
    meanpool = table[table["aggregator"] == "meanpool"]
    retuned = meanpool.assign(run_id=meanpool["run_id"] + "-retuned", learning_rate=1e-3)
    with pytest.raises(ValueError, match="several models"):
        family_predictions(pd.concat([table, retuned]), "aggregator_comparison", folds_dir)


def test_family_refuses_an_empty_selection(toy_table):
    table, folds_dir = toy_table
    without_encoders = table[table["families"] != "encoder_comparison"]
    with pytest.raises(ValueError, match="no five-fold test predictions"):
        family_predictions(without_encoders, "encoder_comparison", folds_dir)


def test_family_refuses_an_incomplete_fold_set(toy_table):
    table, folds_dir = toy_table
    partial = table[~((table["aggregator"] == "meanpool") & (table["fold"] == 1))]
    with pytest.raises(ValueError, match="complete fold set"):
        family_predictions(partial, "aggregator_comparison", folds_dir)


def test_family_refuses_a_second_varying_axis(toy_table):
    table, folds_dir = toy_table
    other_encoder = table["aggregator"] == "meanpool"
    table.loc[other_encoder, "encoder"] = "uni_v2"
    with pytest.raises(ValueError, match="more than its aggregator axis"):
        family_predictions(table, "aggregator_comparison", folds_dir)


def test_paired_difference_sign_and_interval():
    scored = scored_frame({"reference": 0.15, "worse": 0.6})
    result = paired_differences(scored, "reference", n_bootstrap=200, seed=1)
    macro = result[result["finding"] == "macro"].iloc[0]
    assert macro["difference_auroc"] < 0
    assert macro["difference_auroc_ci_high"] < 0
    assert len(result) == len(LUNG) + 1
    finding_rows = result[result["finding"] != "macro"]
    assert np.isclose(finding_rows["difference_auroc"].mean(), macro["difference_auroc"])


def test_identical_variant_has_zero_difference():
    scored = scored_frame({"reference": 0.2})
    twin = scored.copy()
    twin["variant"] = "twin"
    twin["run_id"] = twin["run_id"].str.replace("reference", "twin")
    result = paired_differences(pd.concat([scored, twin]), "reference", n_bootstrap=50, seed=2)
    assert (result["difference_auroc"] == 0).all()
    assert (result["difference_auroc_ci_low"] == 0).all()


def test_missing_reference_raises():
    scored = scored_frame({"a": 0.2})
    with pytest.raises(ValueError):
        paired_differences(scored, "reference", n_bootstrap=10)


def write_counts(runs_dir, run_id, cases, positives):
    """Write a toy training-set counts file of one lung run."""
    labels = {finding: {"pos": positives, "neg": 2 * cases - positives, "masked": 0}
              for finding in LUNG}
    train = {"slides": 2 * cases, "cases": cases, "labels": labels}
    directory = runs_dir / run_id
    directory.mkdir(parents=True)
    (directory / "data_counts.json").write_text(json.dumps({"train": train}))


def test_training_counts_reads_one_row_per_run_and_finding(tmp_path):
    write_counts(tmp_path, "toy-run", cases=30, positives=7)
    counts = training_counts(["toy-run"], tmp_path)
    assert len(counts) == len(LUNG)
    assert set(counts["training_cases"]) == {30}
    assert set(counts["training_slides"]) == {60}
    assert set(counts["training_positives"]) == {7}


def test_learning_curve_summary_averages_training_counts_over_folds(tmp_path):
    scored = scored_frame({0.1: 0.5, 1.0: 0.2})
    for run_id in scored["run_id"].unique():
        fraction, fold = run_id.split("_")
        cases = int(float(fraction) * 100) + int(fold)
        write_counts(tmp_path, run_id, cases=cases, positives=int(fold) + 1)
    counts = training_counts(sorted(scored["run_id"].unique()), tmp_path)
    table = learning_curve_summary(scored, counts)
    macro = table[table["finding"] == "macro"].set_index("fraction")
    assert macro.loc[1.0, "auroc_mean"] > macro.loc[0.1, "auroc_mean"]
    assert macro.loc[0.1, "training_cases"] == 12
    assert macro.loc[0.1, "training_slides"] == 24
    assert macro["training_positives"].isna().all()
    squames = table[table["finding"] == "aspiration_squames"].set_index("fraction")
    assert squames.loc[1.0, "training_positives"] == 3
    assert set(variant_summaries(scored)["variant"]) == {0.1, 1.0}
    assert set(learning_curve_folds(scored)["fraction"]) == {0.1, 1.0}


def test_learning_curve_summary_refuses_a_run_without_counts(tmp_path):
    scored = scored_frame({0.1: 0.5})
    write_counts(tmp_path, "0.1_0", cases=10, positives=5)
    counts = training_counts(["0.1_0"], tmp_path)
    with pytest.raises(ValueError, match="no training counts"):
        learning_curve_summary(scored, counts)
