"""Score every registered model on its validation and test slides."""

import argparse
import json
from pathlib import Path

from champs_pipeline.eval import run_record
from champs_pipeline.eval.inference import prepare_registry, write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    commit, dirty = run_record.git_state(repo)
    if not commit or dirty is not False:
        raise SystemExit("production inference requires a clean committed source tree")
    contexts = prepare_registry(args.registry, args.runs_dir, args.folds_dir, args.matrix)
    if args.preflight:
        print(json.dumps({"preflight": "pass", "runs": len(contexts), "git_commit": commit}))
        return
    record = write_predictions(contexts, args.out, repo, args.registry, args.matrix, args.device)
    print(json.dumps({"status": record["status"], "artifact_sha256": record["artifact_sha256"]}))


if __name__ == "__main__":
    main()
