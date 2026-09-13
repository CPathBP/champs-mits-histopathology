"""Case-level fold designs on a manifest, and the class-count gate.

Two designs share the balanced microfold assignment of ``fivefold``:

- ``fivefold``: ten case microfolds; per outer fold two are the test set,
  one the validation set, seven the training set.
- ``loso_nested``: one fold per site; the site is the test set and one
  tenth of the other sites' cases, balanced the same way, the validation
  set.
"""

import numpy as np
import pandas as pd

from champs_pipeline.data_prep.fivefold import (
    FoldDesign, assign_balanced_microfolds, build_case_table, split_assignments,
)

SPLITS = ("train", "val", "test")


def fivefold_case_map(manifest, labels, seed):
    """Per case: site, slide count, the labels, and the microfold of the five-fold design."""
    design = FoldDesign()
    cases = build_case_table(manifest, labels)
    cases["microfold"] = assign_balanced_microfolds(cases, labels, n_microfolds=design.n_microfolds,
                                                    seed=seed)
    return cases.sort_values("case_id").reset_index(drop=True)


def fivefold_splits(case_map):
    """Long table of (fold, case_id, split) for the five outer folds."""
    design = FoldDesign()
    rows = []
    for fold in range(design.n_outer_folds):
        split = split_assignments(case_map, fold, design=design)
        rows.append(pd.DataFrame({"fold": fold, "case_id": split.index, "split": split.values}))
    return pd.concat(rows, ignore_index=True)


def site_splits(manifest, labels, seed, val_fraction=0.1):
    """Long table of (fold, site, case_id, split) with one fold per site.

    The held-out site is the test set. One microfold of the remaining
    cases, balanced on site and labels, is the validation set.
    """
    n_microfolds = int(round(1.0 / val_fraction))
    cases = build_case_table(manifest, labels)
    rows = []
    for fold, site in enumerate(sorted(cases["case_location"].unique())):
        pool = cases[cases["case_location"] != site].reset_index(drop=True)
        microfold = assign_balanced_microfolds(pool, labels, n_microfolds=n_microfolds,
                                               seed=seed + fold)
        split = np.where(microfold == 0, "val", "train")
        rows.append(pd.DataFrame({"fold": fold, "site": site, "case_id": pool["case_id"],
                                  "split": split}))
        held_out = cases.loc[cases["case_location"] == site, "case_id"]
        rows.append(pd.DataFrame({"fold": fold, "site": site, "case_id": held_out,
                                  "split": "test"}))
    return pd.concat(rows, ignore_index=True)


def fold_manifest(manifest, splits, fold):
    """The manifest with a ``split`` column for one fold; refuses a case in two splits."""
    assignment = splits[splits["fold"] == fold].set_index("case_id")["split"]
    out = manifest.drop(columns=["split"], errors="ignore").copy()
    out["split"] = out["case_id"].astype(str).map(assignment)
    if out["split"].isna().any():
        raise ValueError(f"fold {fold}: a slide has no split")
    if out.groupby("case_id")["split"].nunique().gt(1).any():
        raise ValueError(f"fold {fold}: a case is in two splits")
    return out


def check_test_coverage(splits):
    """Every case is held out exactly once across the folds of a design."""
    held = splits[splits["split"] == "test"].groupby("case_id").size()
    cases = set(splits["case_id"])
    if set(held.index) != cases or (held != 1).any():
        raise ValueError("cases are not held out exactly once")


def class_counts(fold_df, label, split):
    """Kept positives, kept negatives and masked slides of one label in one split."""
    rows = fold_df[fold_df["split"] == split]
    y = pd.to_numeric(rows[f"label_{label}"], errors="raise")
    keep = pd.to_numeric(rows[f"mask_{label}"], errors="raise") == 1.0
    positive = keep & (y == 1.0)
    negative = keep & (y == 0.0)
    return {
        f"{split}_pos": int(positive.sum()), f"{split}_neg": int(negative.sum()),
        f"{split}_masked": int((~keep).sum()),
        f"{split}_pos_cases": int(rows.loc[positive, "case_id"].nunique()),
        f"{split}_neg_cases": int(rows.loc[negative, "case_id"].nunique()),
    }


def gate_status(row, core, site_design):
    """PASS, a WARN, or a FAIL for one (fold, label) cell.

    A training split without both classes always fails. A validation
    split without both classes fails for the findings that select the
    checkpoint and warns otherwise. A held-out site without a class is a
    fact about the site and warns; in the five-fold design a test split
    without a class fails for the core findings.
    """
    if row["train_pos"] == 0 or row["train_neg"] == 0:
        return "FAIL_TRAIN_CLASS"
    if row["val_pos"] == 0 or row["val_neg"] == 0:
        return "FAIL_CORE_VAL_CLASS" if core else "WARN_VAL_NO_CLASS"
    if row["test_pos"] == 0 or row["test_neg"] == 0:
        if core and not site_design:
            return "FAIL_CORE_TEST_CLASS"
        return "WARN_TEST_NO_CLASS"
    if site_design and row["test_pos"] < 10:
        return "WARN_TEST_THIN"
    if row["val_pos"] < 5 or (not site_design and row["test_pos"] < 5):
        return "WARN_THIN_POSITIVE"
    return "PASS"


def gate_table(folds, labels, core_labels, site_design):
    """Class counts and gate status per (fold, label) over the fold manifests given."""
    rows = []
    for fold, fold_df in folds.items():
        for label in labels:
            row = {"fold": fold, "label": label, "is_core": label in core_labels}
            for split in SPLITS:
                row.update(class_counts(fold_df, label, split))
            ratio = row["train_neg"] / row["train_pos"] if row["train_pos"] else float("inf")
            row["train_neg_per_pos"] = round(ratio, 3)
            row["gate_status"] = gate_status(row, label in core_labels, site_design)
            rows.append(row)
    return pd.DataFrame(rows)
