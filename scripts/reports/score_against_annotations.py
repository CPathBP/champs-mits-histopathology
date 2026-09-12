"""Score the extracted findings against human annotations of the report text.

An annotation file (one JSON per case) lists the (organ, text source) cells
a reader reviewed and, per cell, the findings the report asserts, with
severity, and whether the statement was negated or hedged. Over the
reviewed cells, this script scores each finding of the extraction
against the annotation: precision and recall with 95% intervals, severity
agreement on the four-bin ladder, how often the extraction abstained (an
uncertain record without a positive), and the share of cases in which the
extraction made no error on any scored finding. Writes
``per_finding.csv`` and ``summary.json`` into ``--out-dir``.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

PAPER_FINDINGS = [
    "aspiration_squames", "aspiration_meconium", "bronchopneumonia", "pneumonitis",
    "hyaline_membranes", "hemozoin_pigment", "steatosis",
]
# Annotations made under the five-bin ladder map onto the four bins of the schema.
FOUR_BINS = {"minimal": "minimal", "mild": "mild", "moderate": "moderate",
             "severe": "severe", "extensive": "severe"}


def wilson(k, n, z=1.96):
    """Wilson 95% interval of a proportion, as (low, high); None when n is 0."""
    if not n:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 3), round(centre + half, 3)


def load_annotations(directory):
    """Per case: the reviewed cells and, per (cell, finding), the annotation's verdict."""
    cases = {}
    for path in sorted(Path(directory).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "complete":
            continue
        cells = {tuple(cell) for cell in record["annotated_cells"]}
        verdicts = {}
        for f in record["findings"]:
            key = (f["organ"], f["text_source"], f["condition"])
            if f.get("negated"):
                verdicts[key] = ("negated", None)
            elif f.get("uncertain"):
                verdicts[key] = ("uncertain", None)
            else:
                verdicts[key] = ("positive", FOUR_BINS.get(f.get("severity")))
        cases[record["case_id"]] = {"cells": cells, "verdicts": verdicts}
    return cases


def extraction_verdicts(findings, case_ids):
    """Per (case, organ, text source, finding): positive with severity, or abstained."""
    rows = findings[findings["champs_deid"].isin(case_ids)]
    verdicts = {}
    for r in rows[rows["kind"] == "uncertain"].itertuples(index=False):
        verdicts[(r.champs_deid, r.organ, r.text_source, r.semantic_group)] = ("abstained", None)
    for r in rows[rows["kind"] == "condition"].itertuples(index=False):
        severity = FOUR_BINS.get(r.severity) if isinstance(r.severity, str) else None
        verdicts[(r.champs_deid, r.organ, r.text_source, r.semantic_group)] = ("positive", severity)
    return verdicts


def score(annotations, extraction, finding):
    """Counts of one finding over every reviewed cell, and the cases with an error."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "abstained": 0, "gold_uncertain": 0,
              "severity_agree": 0, "severity_n": 0}
    cases_with_error = set()
    for case_id, annotation in annotations.items():
        for organ, source in annotation["cells"]:
            verdicts = annotation["verdicts"]
            gold, gold_severity = verdicts.get((organ, source, finding), ("absent", None))
            got, got_severity = extraction.get((case_id, organ, source, finding), ("absent", None))
            if gold == "uncertain":
                counts["gold_uncertain"] += 1
                continue
            if got == "abstained":
                counts["abstained"] += 1
                continue
            gold_positive = gold == "positive"
            got_positive = got == "positive"
            if gold_positive and got_positive:
                counts["tp"] += 1
                if gold_severity and got_severity:
                    counts["severity_n"] += 1
                    counts["severity_agree"] += gold_severity == got_severity
            elif got_positive:
                counts["fp"] += 1
                cases_with_error.add(case_id)
            elif gold_positive:
                counts["fn"] += 1
                cases_with_error.add(case_id)
            else:
                counts["tn"] += 1
    return counts, cases_with_error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--findings", required=True)
    ap.add_argument("--annotations", required=True, help="Directory of per-case annotation JSON.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--all-findings", action="store_true",
                    help="Score every finding the annotations or the extraction name.")
    args = ap.parse_args()

    annotations = load_annotations(args.annotations)
    findings = pd.read_parquet(args.findings)
    extraction = extraction_verdicts(findings, set(annotations))
    scored = PAPER_FINDINGS
    if args.all_findings:
        named = {key[2] for a in annotations.values() for key in a["verdicts"]}
        named |= {key[3] for key in extraction}
        scored = sorted(named)

    rows = []
    cases_with_error = set()
    for finding in scored:
        counts, erring = score(annotations, extraction, finding)
        cases_with_error |= erring
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision_ci = wilson(tp, tp + fp)
        recall_ci = wilson(tp, tp + fn)
        rows.append({
            "finding": finding,
            "n_cells": sum(len(a["cells"]) for a in annotations.values()),
            "gold_positive": tp + fn,
            "extracted_positive": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": counts["tn"],
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "precision_ci_low": precision_ci[0], "precision_ci_high": precision_ci[1],
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "recall_ci_low": recall_ci[0], "recall_ci_high": recall_ci[1],
            "severity_agreement": (round(counts["severity_agree"] / counts["severity_n"], 3)
                                   if counts["severity_n"] else None),
            "severity_n": counts["severity_n"],
            "abstained": counts["abstained"],
            "gold_uncertain": counts["gold_uncertain"],
        })
    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "per_finding.csv", index=False)
    tp_all = int(table["tp"].sum())
    summary = {
        "n_cases": len(annotations),
        "n_cells": int(sum(len(a["cells"]) for a in annotations.values())),
        "findings_scored": scored,
        "micro_precision": round(tp_all / max(tp_all + table["fp"].sum(), 1), 4),
        "micro_recall": round(tp_all / max(tp_all + table["fn"].sum(), 1), 4),
        "cases_without_error": len(annotations) - len(cases_with_error),
        "case_level_accuracy": round(1 - len(cases_with_error) / len(annotations), 4),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
