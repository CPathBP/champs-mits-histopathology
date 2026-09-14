"""Nested, prevalence-preserving subsets of the training cases.

A learning curve trains on growing fractions of the training set. Every
fraction is a prefix of one seeded case order whose every prefix keeps
each finding's prevalence, so the curve is not confounded by prevalence
drift among the rarer findings. Cases are keyed by their rarest positive
finding, shuffled within the key, and spread evenly over the order
(systematic interleave), in the spirit of iterative stratification
(Sechidis et al., 2011).
"""

import math

import numpy as np


def case_order(frame, labels, case_col="case_id", seed=42):
    """The seeded order of the cases of ``frame``; every prefix preserves prevalence."""
    rng = np.random.default_rng(seed)
    columns = [f"label_{label}" for label in labels]
    per_case = frame.groupby(case_col)[columns].max().fillna(0.0).astype(int)
    per_case.index = per_case.index.astype(str)
    if not columns:
        order = list(per_case.index)
        rng.shuffle(order)
        return order

    rank = {column: r for r, column in enumerate(per_case.mean().sort_values().index)}

    def key(row):
        present = [column for column in columns if row[column] == 1]
        if not present:
            return "__none__"
        return min(present, key=lambda column: rank[column])

    groups = {}
    for case_id, group in zip(per_case.index, per_case.apply(key, axis=1)):
        groups.setdefault(group, []).append(case_id)
    for members in groups.values():
        rng.shuffle(members)
    placed = []
    for members in groups.values():
        for i, case_id in enumerate(members):
            placed.append(((i + 0.5) / len(members), float(rng.random()), case_id))
    placed.sort(key=lambda item: (item[0], item[1]))
    return [case_id for _, _, case_id in placed]


def subsample_train_by_fraction(frame, fraction, labels, case_col="case_id", seed=42):
    """The rows of the first ``ceil(fraction * n_cases)`` cases of the seeded order."""
    if fraction is None or fraction >= 1.0:
        return frame
    order = case_order(frame, labels, case_col=case_col, seed=seed)
    keep = set(order[: max(1, math.ceil(fraction * len(order)))])
    return frame[frame[case_col].astype(str).isin(keep)]
