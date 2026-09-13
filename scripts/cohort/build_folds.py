"""Build the case splits, the fold manifests and the class-count gate.

Per organ group, three designs on the manifests: ``fivefold`` (balanced
case microfolds, seed 42, shared by every label variant),
``loso_nested`` (one fold per site with an inner validation split) and
``intersect_fivefold`` (the five-fold design over the slides every
compared encoder covers). Writes ``case_splits.csv`` (organ, design,
fold, case, split), ``fold_csvs/<organ>_<variant>_<design>/fold_<k>.csv``
(the manifest with a ``split`` column, what training reads),
``gate_table.csv`` and ``gate_report.md``. Exits with an error when a
training split or a core validation split lacks a class.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from champs_pipeline.data_prep.fivefold import validate_manifest_compatibility
from champs_pipeline.data_prep.folds import (
    check_test_coverage, fivefold_case_map, fivefold_splits, fold_manifest, gate_table,
    site_splits,
)
from champs_pipeline.data_prep.reference_a import VARIANTS

SITE_DESIGN = "loso_nested"


def covered_by_every_encoder(manifest, encoders):
    """The manifest rows with a feature file for every compared encoder."""
    keep = pd.Series(True, index=manifest.index)
    for encoder in encoders:
        keep &= manifest[f"feature_path_{encoder}"].notna()
    return manifest[keep].reset_index(drop=True)


def designs_for(manifest, labels, seed, comparison_encoders):
    """The (design, splits, slide subset) triples of one organ group.

    The intersection design is built when ``comparison_encoders`` is not
    empty (the encoder comparison runs on one organ group).
    """
    case_map = fivefold_case_map(manifest, labels, seed)
    designs = [
        ("fivefold", fivefold_splits(case_map), None),
        (SITE_DESIGN, site_splits(manifest, labels, seed), None),
    ]
    if comparison_encoders:
        subset = covered_by_every_encoder(manifest, comparison_encoders)
        subset_map = fivefold_case_map(subset, labels, seed)
        designs.append(("intersect_fivefold", fivefold_splits(subset_map), set(subset["slide_id"])))
    return designs


def markdown_table(frame):
    """The frame as a Markdown table."""
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + body)


def write_report(table, out_path):
    """A short Markdown summary of the gate: counts per organ and design, then flagged cells."""
    counts = []
    for (organ, design, variant), part in table.groupby(["organ", "design", "variant"]):
        status = part["gate_status"]
        counts.append({"organ": organ, "design": design, "variant": variant, "cells": len(part),
                       "pass": int(status.eq("PASS").sum()),
                       "warn": int(status.str.startswith("WARN").sum()),
                       "fail": int(status.str.startswith("FAIL").sum())})
    lines = ["# Class-count gate", "", markdown_table(pd.DataFrame(counts)), "",
             "## Flagged cells", ""]
    flagged = table[table["gate_status"] != "PASS"]
    if flagged.empty:
        lines.append("None.")
    else:
        columns = ["organ", "design", "variant", "fold", "label", "train_pos", "train_neg",
                   "val_pos", "val_neg", "test_pos", "test_neg", "gate_status"]
        lines.append(markdown_table(flagged[columns]))
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, action="append", required=True)
    ap.add_argument("--encoders", required=True, help="configs/cohort/encoders.yaml")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    encoders = yaml.safe_load(Path(args.encoders).read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_parts, gate_parts = [], []
    for path in args.config:
        config = yaml.safe_load(path.read_text())
        organ, labels = config["organ"], config["labels"]
        manifests = {v: pd.read_csv(args.manifest_dir / f"manifest_{organ}_{v}.csv",
                                    low_memory=False, dtype={"case_id": str}) for v in VARIANTS}
        for variant in VARIANTS:
            validate_manifest_compatibility(manifests["elig"], manifests[variant], labels)
        comparison = encoders["comparison_encoders"] if config.get("encoder_comparison") else []
        for design, splits, subset in designs_for(manifests["elig"], labels, args.seed, comparison):
            check_test_coverage(splits)
            split_parts.append(splits.assign(organ=organ, design=design))
            for variant, manifest in manifests.items():
                rows = manifest if subset is None else manifest[manifest["slide_id"].isin(subset)]
                fold_dir = args.out_dir / "fold_csvs" / f"{organ}_{variant}_{design}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                folds = {}
                for fold in sorted(splits["fold"].unique()):
                    folds[fold] = fold_manifest(rows, splits, fold)
                    folds[fold].to_csv(fold_dir / f"fold_{fold}.csv", index=False)
                gate = gate_table(folds, labels, config["core_labels"], design == SITE_DESIGN)
                gate_parts.append(gate.assign(organ=organ, design=design, variant=variant))
                print(f"{organ} {design} {variant}: {len(folds)} folds, "
                      f"{gate['gate_status'].value_counts().to_dict()}")

    case_splits = pd.concat(split_parts, ignore_index=True)
    columns = ["organ", "design", "fold", "site", "case_id", "split"]
    case_splits = case_splits.reindex(columns=columns)
    case_splits.to_csv(args.out_dir / "case_splits.csv", index=False)
    gate = pd.concat(gate_parts, ignore_index=True)
    gate.to_csv(args.out_dir / "gate_table.csv", index=False)
    write_report(gate, args.out_dir / "gate_report.md")
    failures = gate[gate["gate_status"].str.startswith("FAIL")]
    if len(failures):
        raise SystemExit(f"gate failed in {len(failures)} cells; see gate_report.md")
    print("gate passed")


if __name__ == "__main__":
    main()
