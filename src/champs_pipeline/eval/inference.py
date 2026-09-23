"""Selected-checkpoint inference on complete validation and test feature bags."""

import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from champs_pipeline.datasets.wsi.dataset import (SlideBagDataset, check_label_cells,
                                                 store_columns)
from champs_pipeline.eval import load, run_record
from champs_pipeline.eval.predictions import (AXES, CELL_COLUMNS, check_cells, record_path,
                                              sha256, validate_predictions)
from champs_pipeline.models.mil.module import MILModule

MATRIX_COLUMNS = ["run_id", "families", "organ", "variant", "design", "fold", "encoder",
                  "aggregator", "learning_rate", "train_fraction", "seed"]
SPLITS = ("val", "test")


def read_json(path):
    return json.loads(Path(path).read_text())


def metadata(row):
    """Spell matrix axes consistently with the canonical prediction contract."""
    return {name: row[{"organ_group": "organ", "label_variant": "variant"}.get(name, name)]
            for name in AXES}


def store_identity(path):
    """Identify the current Lance snapshot without hashing the full feature corpus."""
    store = lance.dataset(str(path))
    manifests = sorted((Path(path) / "_versions").glob("*.manifest"))
    return {"version": int(store.version), "rows": store.count_rows(),
            "manifests": {p.name: sha256(p) for p in manifests}}


def _fold_parts(path, labels, encoder):
    """Validate identities and Reference A before selecting held-out splits."""
    frame = pd.read_csv(path, low_memory=False, dtype={"slide_id": str, "case_id": str})
    path_col, row_col = store_columns(encoder)
    required = ["slide_id", "case_id", "split", path_col, row_col]
    required += [f"{prefix}{label}" for prefix in ("label_", "mask_") for label in labels]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"fold lacks columns: {sorted(missing)}")
    frame = frame[required].copy()
    if frame[["slide_id", "case_id", "split"]].isna().any().any():
        raise ValueError("fold has missing slide/case/split identities")
    if frame.duplicated("slide_id").any() or set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError("fold has duplicate slides or invalid splits")
    if frame.groupby("case_id")["split"].nunique().gt(1).any():
        raise ValueError("fold leaks a case across splits")
    check_label_cells(frame, labels, "mask_", path)
    for label in labels:
        if frame.loc[frame[f"mask_{label}"].eq(0), f"label_{label}"].notna().any():
            raise ValueError("masked labels must be null")
    parts = {split: frame.loc[frame["split"].eq(split)].copy() for split in SPLITS}
    for split, part in parts.items():
        if part.empty or part[[path_col, row_col]].isna().any().any():
            raise ValueError(f"{split} lacks features; explicit upstream adjudication required")
        indices = pd.to_numeric(part[row_col], errors="raise")
        if not np.isfinite(indices).all() or (indices < 0).any() or indices.mod(1).ne(0).any():
            raise ValueError("invalid Lance row index")
        for store_path, group in part.groupby(path_col):
            store = lance.dataset(store_path)
            actual = store.take(group[row_col].astype(int).tolist(), columns=["slide_id"])
            if actual["slide_id"].to_pylist() != group["slide_id"].tolist():
                raise ValueError("Lance row holds another slide")
    return parts


def prepare_run(row, runs_dir, folds_dir, allow_dirty=False):
    """Bind one registry row to unchanged source files and checkpoint metadata."""
    row = dict(row)
    run_id = row["run_id"]
    if not isinstance(run_id, str) or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("invalid run id")
    directory = Path(runs_dir) / run_id
    record = run_record.read(directory)
    if not record or record.get("status") != "complete":
        raise ValueError(f"{run_id}: run is not complete")
    if not allow_dirty and (record.get("git_dirty") is not False or not record.get("git_commit")):
        raise ValueError(f"{run_id}: training has no clean commit")
    identity = MATRIX_COLUMNS + ["git_commit", "git_dirty", "selected_checkpoint", "selected_epoch"]
    for key in identity:
        wanted = record["matrix_row"].get(key) if key in MATRIX_COLUMNS else record.get(key)
        if key not in row or row[key] != wanted:
            raise ValueError(f"{run_id}: registry {key} differs from run record")
    if row.get("status", "complete") != "complete":
        raise ValueError(f"{run_id}: registry run is not complete")
    checkpoint = directory / record["selected_checkpoint"]
    if not checkpoint.resolve().is_relative_to(directory.resolve()):
        raise ValueError("checkpoint is outside its run directory")
    fold = Path(folds_dir) / f"{row['organ']}_{row['variant']}_{row['design']}"
    fold = fold / f"fold_{int(row['fold'])}.csv"
    paths = {"run_record": directory / "run.json",
             "label_mapping": directory / "label_mapping.json",
             "hparams": directory / "hparams.yaml", "checkpoint": checkpoint, "fold_csv": fold}
    inputs = {name: sha256(path) for name, path in paths.items()}
    if inputs["fold_csv"] != record.get("inputs", {}).get("fold_csv"):
        raise ValueError(f"{run_id}: fold hash differs from training input")
    mapping = read_json(paths["label_mapping"])
    indexed = mapping["idx_to_label"]
    if set(indexed) != {str(i) for i in range(len(indexed))} or not indexed:
        raise ValueError("label mapping indices must be contiguous from zero")
    labels = [indexed[str(i)] for i in range(len(indexed))]
    if len(set(labels)) != len(labels) or not all(isinstance(v, str) and v for v in labels):
        raise ValueError("invalid finding names in label mapping")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    if hp != yaml.safe_load(paths["hparams"].read_text()):
        raise ValueError("checkpoint hyperparameters differ from hparams.yaml")
    if (hp["labels"] != labels or hp["selection_labels"] != mapping["selection_labels"]
            or hp["aggregator"] != row["aggregator"]
            or hp["learning_rate"] != row["learning_rate"]
            or state["epoch"] != row["selected_epoch"]):
        raise ValueError("checkpoint labels, architecture or epoch disagree with run identity")
    parts = _fold_parts(fold, labels, row["encoder"])
    path_col, _ = store_columns(row["encoder"])
    stores = {str(path): store_identity(path) for part in parts.values()
              for path in part[path_col].unique()}
    return {"row": row, "labels": labels, "parts": parts, "checkpoint": checkpoint,
            "hparams": hp, "inputs": inputs, "paths": paths, "stores": stores}


def prepare_registry(registry_path, runs_dir, folds_dir, matrix_path):
    """Refuse missing matrix runs and prepare every entry before execution."""
    registry = pd.read_csv(registry_path)
    matrix = pd.read_csv(matrix_path)
    for frame in (registry, matrix):
        if frame.empty or frame["run_id"].isna().any() or frame.duplicated("run_id").any():
            raise ValueError("matrix/registry has missing or duplicate run ids")
    left = registry.set_index("run_id")[MATRIX_COLUMNS[1:]].sort_index()
    right = matrix.set_index("run_id")[MATRIX_COLUMNS[1:]].sort_index()
    if not left.equals(right):
        raise ValueError("registry differs from the complete training matrix")
    return [prepare_run(row, runs_dir, folds_dir) for row in registry.to_dict("records")]


def expected_cells(context):
    """Build cell identities and Reference A directly from fold and head order."""
    frames = []
    for split, part in context["parts"].items():
        for finding in context["labels"]:
            frame = part[["slide_id", "case_id"]].copy()
            for name, value in metadata(context["row"]).items():
                frame[name] = value
            frame["split"], frame["finding"] = split, finding
            frame["y_true"] = part[f"label_{finding}"]
            frame["mask"] = part[f"mask_{finding}"].astype(int)
            frames.append(frame[CELL_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def verify_inputs(context):
    """Refuse source files or feature versions changed since preflight."""
    for name, path in context["paths"].items():
        if sha256(path) != context["inputs"][name]:
            raise ValueError(f"inference input changed: {name}")
    for path, identity in context["stores"].items():
        if store_identity(path) != identity:
            raise ValueError(f"feature store changed: {path}")


def predict_run(context, device="cpu"):
    """Score full bags in float32, retaining null labels and every output head."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    verify_inputs(context)
    hp = context["hparams"]
    module = MILModule(**hp)
    state = torch.load(context["checkpoint"], map_location="cpu", weights_only=False)
    module.load_state_dict(state["state_dict"], strict=True)
    model = module.model.float().to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    axes = metadata(context["row"])
    for split, part in context["parts"].items():
        dataset = SlideBagDataset(part, context["labels"], context["row"]["encoder"])
        for index in range(len(dataset)):
            features = dataset[index]["features"]
            if (features.ndim != 2 or not len(features) or features.shape[1] != hp["input_dim"]
                    or not torch.isfinite(features).all()):
                raise ValueError("invalid feature bag or embedding width")
            with torch.inference_mode():
                logits = model(features.to(device=device, dtype=torch.float32))["logits"]
                if logits.shape != (1, len(context["labels"])):
                    raise ValueError("model output does not match recorded finding heads")
                logits = logits[0].float().cpu()
                scores = logits.sigmoid().numpy()
            source = part.iloc[index]
            for head, finding in enumerate(context["labels"]):
                rows.append({**axes, "split": split, "slide_id": source["slide_id"],
                             "case_id": source["case_id"], "finding": finding,
                             "y_true": source[f"label_{finding}"],
                             "mask": int(source[f"mask_{finding}"]),
                             "score": scores[head], "logit": logits.numpy()[head]})
    result = validate_predictions(pd.DataFrame(rows))
    check_cells(result, expected_cells(context))
    verify_inputs(context)
    return result


def check_artifact(path, contexts):
    """Check the serialized artifact against independent source cells."""
    expected = pd.concat([expected_cells(c) for c in contexts], ignore_index=True)
    return load.predictions(path).check(expected).frame


def write_predictions(contexts, out, repo, registry_path, matrix_path, device="cpu"):
    """Publish a checked parquet and hash-bound record; partial files fail closed."""
    commit, dirty = run_record.git_state(repo)
    if dirty is not False or not commit:
        raise ValueError("production inference requires a clean committed source tree")
    source_hashes = {"registry": sha256(registry_path), "matrix": sha256(matrix_path)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = run_record.now()
    descriptor, name = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
    os.close(descriptor)
    temporary = Path(name)
    record_temp = Path(name + ".json")
    try:
        writer = None
        try:
            for context in contexts:
                table = pa.Table.from_pandas(predict_run(context, device), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        frame = check_artifact(temporary, contexts)
        for context in contexts:
            verify_inputs(context)
        if (source_hashes != {"registry": sha256(registry_path), "matrix": sha256(matrix_path)}
                or run_record.git_state(repo) != (commit, False)):
            raise ValueError("source or registry changed during inference")
        packages = {name: importlib.metadata.version(name) for name in
                    ("torch", "pandas", "numpy", "pyarrow", "pylance", "pytorch-lightning")}
        record = {"schema_version": 1, "status": "complete", "started": started,
                  "finished": run_record.now(), "git_commit": commit, "git_dirty": False,
                  "device": device, "precision": "float32", "splits": list(SPLITS),
                  "python": platform.python_version(), "packages": packages,
                  "source_hashes": source_hashes, "artifact_sha256": sha256(temporary),
                  "cells_by_run": {str(k): int(v)
                                   for k, v in frame.groupby("run_id").size().items()},
                  "runs": {c["row"]["run_id"]: {"inputs": c["inputs"], "stores": c["stores"],
                           "training_commit": c["row"]["git_commit"], "labels": c["labels"]}
                           for c in contexts}}
        record_temp.write_text(json.dumps(record, indent=2) + "\n")
        # A crash between replacements leaves a hash mismatch, never false completeness.
        os.replace(temporary, out)
        os.replace(record_temp, record_path(out))
    finally:
        temporary.unlink(missing_ok=True)
        record_temp.unlink(missing_ok=True)
    return record


def check_provenance(path, contexts, registry_path, matrix_path):
    """Bind a checked artifact to the exact registry and checkpoints now supplied."""
    record = read_json(record_path(path))
    sources = {"registry": sha256(registry_path), "matrix": sha256(matrix_path)}
    if record.get("source_hashes") != sources:
        raise ValueError("prediction provenance differs from the supplied registry or matrix")
    expected = {c["row"]["run_id"]: {"inputs": c["inputs"], "stores": c["stores"],
                "training_commit": c["row"]["git_commit"], "labels": c["labels"]}
                for c in contexts}
    if record.get("runs") != expected:
        raise ValueError("prediction provenance differs from the supplied run inputs")
    for context in contexts:
        verify_inputs(context)
