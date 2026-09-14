"""Expand the training matrix configuration into one row per run.

Writes ``training_matrix.csv``: run id, families, organ, label variant,
fold design, fold, encoder, aggregator, learning rate, training fraction
and seed.
"""

import argparse
from pathlib import Path

import yaml

from champs_pipeline.training.matrix import count_folds, expand

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-config", type=Path, default=REPO / "configs/training/matrix.yaml")
    ap.add_argument("--models-dir", type=Path, default=REPO / "configs/training/models")
    ap.add_argument("--folds-dir", type=Path, required=True,
                    help="The fold_csvs directory of the cohort stage.")
    ap.add_argument("--families", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    matrix = yaml.safe_load(args.matrix_config.read_text())
    models = {path.stem: yaml.safe_load(path.read_text())
              for path in sorted(args.models_dir.glob("*.yaml"))}

    def n_folds(organ, variant, design):
        return count_folds(args.folds_dir, organ, variant, design)

    rows = expand(matrix, models, n_folds, families=args.families)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out, index=False)
    per_family = rows["families"].str.split(";").explode().value_counts().sort_index()
    print(f"{len(rows)} runs\n{per_family.to_string()}")


if __name__ == "__main__":
    main()
