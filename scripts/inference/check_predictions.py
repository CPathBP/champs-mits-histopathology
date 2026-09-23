"""Check canonical predictions against every registered source cell."""

import argparse
from pathlib import Path

from champs_pipeline.eval import load
from champs_pipeline.eval.inference import check_artifact, check_provenance, prepare_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    contexts = prepare_registry(args.registry, args.runs_dir, args.folds_dir, args.matrix)
    frame = check_artifact(args.predictions, contexts)
    load.predictions(args.predictions).check()
    check_provenance(args.predictions, contexts, args.registry, args.matrix)
    print(f"complete: {len(contexts)} runs, {len(frame)} prediction cells")


if __name__ == "__main__":
    main()
