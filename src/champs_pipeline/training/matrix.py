"""The training matrix and the run registry.

The matrix configuration names families of runs by the axes they vary.
Expanded, it is one row per run with a stable run id; a run that two
families share appears once, listing both. The registry joins every
matrix row to what its run directory says about itself.
"""

import itertools
from pathlib import Path

import pandas as pd

from champs_pipeline.eval import run_record

AXES = ("organ", "variant", "design", "encoder", "aggregator", "train_fraction", "seed")
COLUMNS = ["run_id", "families", "organ", "variant", "design", "fold", "encoder", "aggregator",
           "learning_rate", "train_fraction", "seed"]
REGISTRY_FIELDS = ["status", "git_commit", "git_dirty", "slurm_job_id", "started", "finished",
                   "epochs_run", "selected_epoch", "selected_checkpoint", "val_ap_core_macro"]


def run_id(row):
    """A readable id that differs whenever any axis of the run differs."""
    return (f"{row['organ']}-{row['variant']}-{row['design']}-fold{row['fold']}-{row['encoder']}"
            f"-{row['aggregator']}-frac{row['train_fraction']:.2f}-seed{row['seed']}")


def fold_dir(folds_dir, organ, variant, design):
    return Path(folds_dir) / f"{organ}_{variant}_{design}"


def count_folds(folds_dir, organ, variant, design):
    """The number of fold manifests of one fold set; refuses an empty or missing set."""
    directory = fold_dir(folds_dir, organ, variant, design)
    n_folds = len(list(directory.glob("fold_*.csv")))
    if n_folds == 0:
        raise FileNotFoundError(f"no fold manifests in {directory}")
    return n_folds


def as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def expand(matrix, models, n_folds, families=None):
    """One row per run of the named families (all when ``families`` is None).

    ``models`` maps an aggregator to its model configuration (for the
    learning rate); ``n_folds(organ, variant, design)`` gives the fold count
    of a fold set.
    """
    rows = {}
    for family, spec in matrix["families"].items():
        if families is not None and family not in families:
            continue
        axes = {axis: as_list(spec.get(axis, matrix["defaults"][axis])) for axis in AXES}
        for combination in itertools.product(*axes.values()):
            fields = dict(zip(axes, combination))
            for fold in range(n_folds(fields["organ"], fields["variant"], fields["design"])):
                row = {**fields, "fold": fold, "train_fraction": float(fields["train_fraction"]),
                       "seed": int(fields["seed"]),
                       "learning_rate": float(models[fields["aggregator"]]["learning_rate"])}
                key = run_id(row)
                if key not in rows:
                    rows[key] = {"run_id": key, "families": [], **row}
                elif any(rows[key][field] != value for field, value in row.items()):
                    raise ValueError(f"run id {key} names two different runs")
                rows[key]["families"].append(family)
    frame = pd.DataFrame(list(rows.values()), columns=COLUMNS)
    frame["families"] = frame["families"].map(";".join)
    return frame


def registry(matrix_rows, runs_dir):
    """The matrix rows with the state of their run directories (``missing`` when absent)."""
    records = []
    for row in matrix_rows.to_dict("records"):
        record = run_record.read(Path(runs_dir) / row["run_id"]) or {"status": "missing"}
        records.append({**row, **{field: record.get(field) for field in REGISTRY_FIELDS}})
    return pd.DataFrame(records, columns=list(matrix_rows.columns) + REGISTRY_FIELDS)
