"""Define the extraction corpus and deal it into shards.

The corpus is every case whose report carries H&E text after HTML
stripping. The cases are shuffled with a fixed seed and dealt round-robin
into ``--n-shards`` lists, one per extraction client. Writes
``corpus_cases.txt`` (the sorted corpus), ``shard_<i>_of_<n>.txt`` and
``manifest.json`` into ``--out-dir``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from champs_pipeline.data_prep.reports import strip_hetext_html


def corpus_cases(reports_csv):
    """Sorted case ids with non-empty H&E text in any of their report rows."""
    reports = pd.read_csv(reports_csv)
    reports = reports[reports["champs_deid"].notna()]
    has_text = reports["HEText"].apply(strip_hetext_html).astype(bool)
    has_text = has_text.groupby(reports["champs_deid"]).any()
    return sorted(has_text[has_text].index.astype(str))


def deal(cases, n_shards, seed):
    """Shuffle with NumPy's global seed, then deal round-robin into shards."""
    order = np.array(cases, dtype=object)
    np.random.seed(seed)
    np.random.shuffle(order)
    return [list(order[i::n_shards]) for i in range(n_shards)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-csv", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-shards", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cases = corpus_cases(args.reports_csv)
    shards = deal(cases, args.n_shards, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "corpus_cases.txt").write_text("\n".join(cases) + "\n")
    width = len(str(args.n_shards))
    entries = []
    for i, shard in enumerate(shards):
        path = args.out_dir / f"shard_{i:0{width}d}_of_{args.n_shards}.txt"
        path.write_text("\n".join(shard) + "\n")
        entries.append({"index": i, "path": path.name, "n_cases": len(shard)})
    manifest = {
        "n_shards": args.n_shards,
        "total_cases": len(cases),
        "seed": args.seed,
        "shard_sizes": [len(s) for s in shards],
        "shards": entries,
        "reports_csv": Path(args.reports_csv).name,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(cases)} cases in {args.n_shards} shards under {args.out_dir}")


if __name__ == "__main__":
    main()
