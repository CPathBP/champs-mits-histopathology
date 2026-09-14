"""Join the training matrix to the run directories.

Writes ``run_registry.csv``: every matrix row with its status (missing,
running, failed, complete), the commit and dirty flag of the code that ran
it, its job id, start and end, the epochs run, and the selected checkpoint
with its validation macro average precision over the selection findings.
With ``--require-complete`` it exits non-zero when a run is not complete or
a complete run started from a dirty or unknown code state; otherwise it
reports those runs and exits zero.
"""

import argparse
from pathlib import Path

import pandas as pd

from champs_pipeline.training.matrix import registry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()

    runs = registry(pd.read_csv(args.matrix), args.runs_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out, index=False)
    print(runs["status"].value_counts().to_string())

    complete = runs[runs["status"] == "complete"]
    dirty = complete[complete["git_dirty"].map(lambda value: value is not False)]
    problems = []
    if not dirty.empty:
        problems.append(f"{len(dirty)} complete runs have no clean commit, "
                        f"for example {dirty['run_id'].iloc[0]}")
    if len(complete) < len(runs):
        problems.append(f"{len(runs) - len(complete)} runs are not complete")
    for problem in problems:
        print(problem)
    if args.require_complete and problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
