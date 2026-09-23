"""Canonical validation/test predictions shared by inference and evaluation."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["run_id", "split", "slide_id", "finding"]
AXES = ["run_id", "families", "organ_group", "label_variant", "design", "fold",
        "encoder", "aggregator", "learning_rate", "train_fraction", "seed"]
CELL_COLUMNS = AXES + ["slide_id", "case_id", "split", "finding", "y_true", "mask"]
REQUIRED_COLUMNS = CELL_COLUMNS + ["score", "logit"]
STRINGS = [name for name in CELL_COLUMNS if name not in
           {"fold", "learning_rate", "train_fraction", "seed", "y_true", "mask"}]


def sha256(path):
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_path(path):
    return Path(path).with_suffix(".json")


def validate_predictions(predictions):
    """Validate the full table, preserving provenance and split columns."""
    missing = set(REQUIRED_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {sorted(missing)}")
    frame = predictions.loc[:, REQUIRED_COLUMNS].copy()
    if frame.empty or frame.drop(columns="y_true").isna().any().any():
        raise ValueError("predictions are empty or contain missing required values")
    for name in STRINGS:
        if not frame[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"prediction {name} must contain nonempty strings")
    for name in ("fold", "seed", "mask"):
        values = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(values).all() or not values.mod(1).eq(0).all():
            raise ValueError(f"prediction {name} must contain integers")
        frame[name] = values.astype("int64")
    for name in ("learning_rate", "train_fraction", "y_true", "score", "logit"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if name != "y_true" and not np.isfinite(frame[name]).all():
            raise ValueError(f"prediction {name} must be finite")
    allowed = {"split": {"val", "test"}, "design": {"fivefold", "loso_nested"},
               "organ_group": {"lung", "liver"}, "label_variant": {"raw", "elig"},
               "mask": {0, 1}}
    for name, values in allowed.items():
        if set(frame[name]) - values:
            raise ValueError(f"invalid prediction {name}")
    if (frame["fold"] < 0).any() or (frame["learning_rate"] <= 0).any():
        raise ValueError("invalid fold or learning rate")
    if not ((frame["train_fraction"] > 0) & (frame["train_fraction"] <= 1)).all():
        raise ValueError("invalid train fraction")
    if not frame.loc[frame["mask"].eq(1), "y_true"].isin([0, 1]).all():
        raise ValueError("kept labels must be binary and nonnull")
    if frame.loc[frame["mask"].eq(0), "y_true"].notna().any():
        raise ValueError("masked labels must be null")
    if not frame["score"].between(0, 1).all():
        raise ValueError("prediction scores must lie in [0, 1]")
    logits = frame["logit"].to_numpy(dtype=np.float64)
    expected = np.exp(-np.logaddexp(0, -logits))
    if not np.allclose(frame["score"], expected, rtol=1e-6, atol=1e-7):
        raise ValueError("prediction score differs from sigmoid(logit)")
    if frame.duplicated(KEY).any():
        raise ValueError("predictions contain duplicate cells")
    if frame.groupby("run_id")[AXES[1:]].nunique().gt(1).any().any():
        raise ValueError("run axes are inconsistent")
    identities = frame.groupby(["run_id", "slide_id"])[["case_id", "split"]].nunique()
    if identities.gt(1).any().any():
        raise ValueError("a slide has inconsistent case or split identity")
    return frame


def check_cells(frame, expected):
    """Require exact cells and source labels/masks, not merely equal row counts."""
    if expected.empty or expected.duplicated(KEY).any():
        raise ValueError("expected prediction cells are empty or duplicated")
    actual = frame.set_index(KEY).sort_index()
    wanted = expected.set_index(KEY).sort_index()
    if not actual.index.equals(wanted.index):
        missing = len(wanted.index.difference(actual.index))
        extra = len(actual.index.difference(wanted.index))
        raise ValueError(f"prediction coverage differs: {missing} missing, {extra} extra cells")
    for column in CELL_COLUMNS:
        if column in KEY:
            continue
        left, right = actual[column], wanted[column]
        if not (left.eq(right) | (left.isna() & right.isna())).all():
            raise ValueError(f"prediction {column} differs from expected source cells")


def select_test_runs(frame, family, design, label_variant="elig"):
    """Select one experimental family explicitly; refuse accidental model pooling."""
    selected = frame.loc[
        frame["split"].eq("test") & frame["design"].eq(design)
        & frame["label_variant"].eq(label_variant)
        & frame["families"].str.split(";").map(lambda values: family in values)
    ].copy()
    if selected.empty:
        raise ValueError(f"no test predictions for {family}/{design}/{label_variant}")
    if selected.duplicated(["organ_group", "slide_id", "finding"]).any():
        raise ValueError("selected family mixes multiple models for the same test slide")
    return selected


def check_test_folds(frame, folds_dir):
    """Verify selected test cells and complete fold sets using only tabular inputs."""
    if not frame["split"].eq("test").all():
        raise ValueError("fold checks require test rows")
    for (organ, variant, design), group in frame.groupby(
            ["organ_group", "label_variant", "design"]):
        directory = Path(folds_dir) / f"{organ}_{variant}_{design}"
        files = sorted(directory.glob("fold_*.csv"))
        wanted_folds = {int(path.stem.split("_")[-1]) for path in files}
        if not files or set(group["fold"]) != wanted_folds:
            raise ValueError("selected predictions do not cover the complete fold set")
        for _, run in group.groupby("run_id"):
            row = run.iloc[0]
            path = directory / f"fold_{int(row['fold'])}.csv"
            fold = pd.read_csv(path, dtype={"slide_id": str, "case_id": str})
            if fold.duplicated("slide_id").any():
                raise ValueError("fold has duplicate slides")
            part = fold.loc[fold["split"].eq("test")]
            expected = []
            for finding in run["finding"].unique():
                cells = part[["slide_id", "case_id"]].copy()
                for axis in AXES:
                    cells[axis] = row[axis]
                cells["finding"], cells["split"] = finding, "test"
                cells["y_true"] = part[f"label_{finding}"]
                cells["mask"] = part[f"mask_{finding}"].astype(int)
                expected.append(cells[CELL_COLUMNS])
            check_cells(run, pd.concat(expected, ignore_index=True))
