"""Deterministic case-level 5-fold splits for multi-label WSI cohorts.

The construction uses ten balanced case ``microfolds``. Each outer test fold
contains two microfolds (20%), the inner early-stopping validation set contains
one non-test microfold (10%), and the remaining seven are training cases (70%).
Balancing operates jointly on site, case-level multi-label positives, and
site-by-label interactions; slide counts are a secondary balancing term.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldDesign:
    """Fixed 70/10/20 nested split geometry."""

    n_outer_folds: int = 5
    microfolds_per_test: int = 2

    @property
    def n_microfolds(self) -> int:
        return self.n_outer_folds * self.microfolds_per_test

    def microfold_roles(self, outer_fold: int) -> dict[int, str]:
        if not 0 <= outer_fold < self.n_outer_folds:
            raise ValueError(
                f"outer_fold must be in [0, {self.n_outer_folds}), got {outer_fold}"
            )
        first_test = self.microfolds_per_test * outer_fold
        test = {
            (first_test + offset) % self.n_microfolds
            for offset in range(self.microfolds_per_test)
        }
        # The next microfold is inside the current outer-training partition and
        # provides a validation set that is distinct from the locked test set.
        val = (first_test + self.microfolds_per_test) % self.n_microfolds
        return {
            microfold: (
                "test" if microfold in test else "val" if microfold == val else "train"
            )
            for microfold in range(self.n_microfolds)
        }


def label_names(df: pd.DataFrame) -> list[str]:
    """Return binary finding names in manifest column order."""
    labels = [c.removeprefix("label_") for c in df if c.startswith("label_")]
    if not labels:
        raise ValueError("manifest has no label_<finding> columns")
    for label in labels:
        raw = df[f"label_{label}"]
        values = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & values.isna()).any():
            raise ValueError(f"label_{label} contains non-numeric values")
        bad = set(values.unique()) - {0, 1, 0.0, 1.0}
        bad = {value for value in bad if not pd.isna(value)}
        if bad:
            raise ValueError(f"label_{label} is not binary: {sorted(bad)}")
    return labels


def _modal_string(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        raise ValueError("case has no non-null location")
    counts = values.value_counts()
    winners = sorted(counts[counts == counts.max()].index)
    return winners[0]


def build_case_table(df: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """Collapse a slide manifest to one stratification row per case."""
    required = {"slide_id", "case_id", "location"} | {
        f"label_{label}" for label in labels
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    if df["slide_id"].duplicated().any():
        raise ValueError("manifest contains duplicate slide_id rows")
    if df["case_id"].isna().any() or df["case_id"].astype(str).eq("").any():
        raise ValueError("manifest contains missing case_id values")

    aggregations: dict[str, object] = {
        "case_location": ("location", _modal_string),
        "n_slides": ("slide_id", "size"),
    }
    aggregations.update(
        {f"label_{label}": (f"label_{label}", "max") for label in labels}
    )
    cases = (
        df.assign(
            case_id=df["case_id"].astype(str),
            **{
                f"label_{label}": pd.to_numeric(
                    df[f"label_{label}"], errors="raise"
                ).fillna(0)
                for label in labels
            },
        )
        .groupby("case_id", sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    for label in labels:
        cases[f"label_{label}"] = cases[f"label_{label}"].astype(np.int8)
    cases["n_slides"] = cases["n_slides"].astype(int)
    return cases


def _stratification_matrix(
    cases: pd.DataFrame,
    labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Site, label, and site-label feature matrix plus objective weights."""
    sites = sorted(cases["case_location"].unique())
    site = np.column_stack(
        [(cases["case_location"] == name).to_numpy(dtype=float) for name in sites]
    )
    positive = cases[[f"label_{label}" for label in labels]].to_numpy(dtype=float)
    interactions = np.column_stack(
        [site[:, s] * positive[:, label] for s in range(len(sites)) for label in range(len(labels))]
    )
    matrix = np.column_stack([site, positive, interactions])
    # Label marginals receive extra emphasis; site-label interactions keep the
    # positives from being balanced globally while concentrating in one site.
    weights = np.concatenate(
        [
            np.ones(site.shape[1], dtype=float),
            np.full(positive.shape[1], 2.0, dtype=float),
            np.full(interactions.shape[1], 0.5, dtype=float),
        ]
    )
    return matrix, weights


def assign_balanced_microfolds(
    cases: pd.DataFrame,
    labels: Sequence[str],
    *,
    n_microfolds: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Greedily assign cases while matching joint feature and size targets.

    Rare and multi-positive cases are placed first. Assignment minimizes the
    incremental normalized squared error from each microfold's target feature,
    case, and slide counts. Fixed capacities guarantee case counts differ by at
    most one.
    """
    if len(cases) < n_microfolds:
        raise ValueError(
            f"need at least {n_microfolds} cases, got {len(cases)}"
        )
    matrix, weights = _stratification_matrix(cases, labels)
    totals = matrix.sum(axis=0)
    targets = totals / float(n_microfolds)
    normalizers = np.maximum(targets, 0.5)

    rng = np.random.default_rng(seed)
    extra_order = rng.permutation(n_microfolds)
    capacities = np.full(n_microfolds, len(cases) // n_microfolds, dtype=int)
    capacities[extra_order[: len(cases) % n_microfolds]] += 1

    # Positive cases with rare site-label combinations lead the ordering.
    rarity = (matrix * weights / np.maximum(totals, 1.0)).sum(axis=1)
    jitter = rng.random(len(cases))
    order = np.lexsort((jitter, -rarity))
    tie_order = rng.permutation(n_microfolds)
    tie_rank = np.empty(n_microfolds, dtype=int)
    tie_rank[tie_order] = np.arange(n_microfolds)

    feature_counts = np.zeros((n_microfolds, matrix.shape[1]), dtype=float)
    case_counts = np.zeros(n_microfolds, dtype=int)
    slide_counts = np.zeros(n_microfolds, dtype=float)
    target_slides = float(cases["n_slides"].sum()) / n_microfolds
    assignments = np.full(len(cases), -1, dtype=int)

    for case_index in order:
        row = matrix[case_index]
        n_slides = float(cases.iloc[case_index]["n_slides"])
        candidates = np.flatnonzero(case_counts < capacities)
        scores: list[tuple[float, int]] = []
        for microfold in candidates:
            current = feature_counts[microfold]
            feature_delta = np.sum(
                weights
                * (
                    (current + row - targets) ** 2
                    - (current - targets) ** 2
                )
                / normalizers
            )
            slide_delta = (
                (slide_counts[microfold] + n_slides - target_slides) ** 2
                - (slide_counts[microfold] - target_slides) ** 2
            ) / max(target_slides, 1.0)
            # Capacities enforce exact case sizes; this small term encourages
            # smooth filling rather than leaving one bin empty until the end.
            fill = case_counts[microfold] / max(capacities[microfold], 1)
            score = float(feature_delta + 0.25 * slide_delta + 0.05 * fill)
            scores.append((score + 1e-10 * tie_rank[microfold], int(microfold)))
        chosen = min(scores)[1]
        assignments[case_index] = chosen
        feature_counts[chosen] += row
        case_counts[chosen] += 1
        slide_counts[chosen] += n_slides

    if (assignments < 0).any() or not np.array_equal(case_counts, capacities):
        raise RuntimeError("internal error: incomplete microfold assignment")
    return assignments


def split_assignments(
    case_map: pd.DataFrame,
    outer_fold: int,
    *,
    design: FoldDesign = FoldDesign(),
) -> pd.Series:
    """Return case_id-indexed train/val/test roles for one outer fold."""
    roles = design.microfold_roles(outer_fold)
    split = case_map.set_index("case_id")["microfold"].map(roles)
    if split.isna().any():
        raise ValueError("case map contains a microfold outside the design")
    return split


def validate_manifest_compatibility(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    labels: Sequence[str],
) -> None:
    """Require variants to share exactly the slides, cases, sites, and positives."""
    columns = ["slide_id", "case_id", "location"] + [
        f"label_{label}" for label in labels
    ]
    missing = set(columns) - set(candidate.columns)
    if missing:
        raise ValueError(f"candidate manifest is missing columns: {sorted(missing)}")
    left = reference[columns].copy().sort_values("slide_id").reset_index(drop=True)
    right = candidate[columns].copy().sort_values("slide_id").reset_index(drop=True)
    for frame in (left, right):
        frame["slide_id"] = frame["slide_id"].astype(str)
        frame["case_id"] = frame["case_id"].astype(str)
        frame["location"] = frame["location"].astype(str)
        for label in labels:
            frame[f"label_{label}"] = pd.to_numeric(
                frame[f"label_{label}"], errors="raise"
            ).fillna(0.0)
    if not left.equals(right):
        raise ValueError(
            "manifest variants do not share identical slide/case/site/positive labels"
        )


def write_folds(
    manifest: pd.DataFrame,
    case_map: pd.DataFrame,
    out_dir: Path,
    *,
    design: FoldDesign = FoldDesign(),
) -> pd.DataFrame:
    """Write all fold CSVs and return their split-size summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    case_map.to_csv(out_dir / "fivefold_case_map.csv", index=False)
    manifest_cases = set(manifest["case_id"].astype(str))
    mapped_cases = set(case_map["case_id"].astype(str))
    if manifest_cases != mapped_cases:
        raise ValueError("manifest and case map contain different case IDs")

    rows: list[dict[str, int]] = []
    base = manifest.drop(columns=["split"], errors="ignore").copy()
    base["case_id"] = base["case_id"].astype(str)
    for fold in range(design.n_outer_folds):
        case_split = split_assignments(case_map, fold, design=design)
        out = base.copy()
        out["split"] = out["case_id"].map(case_split)
        if out["split"].isna().any():
            raise RuntimeError(f"fold {fold}: at least one slide has no split")
        if out.groupby("case_id")["split"].nunique().gt(1).any():
            raise RuntimeError(f"fold {fold}: case leakage")
        out.to_csv(out_dir / f"fold_{fold}.csv", index=False)

        row: dict[str, int] = {"fold": fold}
        for split in ("train", "val", "test"):
            subset = out[out["split"] == split]
            row[f"{split}_cases"] = int(subset["case_id"].nunique())
            row[f"{split}_slides"] = int(len(subset))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "fivefold_fold_map.csv", index=False)
    return summary
