"""Build the findings table and the case attribution from the extraction records.

Reads ``<per-case>/<case>.json``, checks that ``--schema`` and ``--prompt``
are the ones the run's ``run_meta.json`` records and that every case of
``--corpus`` was extracted or failed, copies the site-report records the
CPL review concurred with, flattens the records into ``findings.parquet``,
and computes ``case_attribution.parquet`` from the report CSV alone.
``case_status.csv`` accounts for every case; ``extraction_record.json``
records the model, the schema and prompt fingerprints, the inference
settings, the case counts, every corrected value, the share of quotes
found verbatim in the report text, and the share found inside the section
the model named.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from champs_pipeline.data_prep.extraction_meta import file_sha, load_run_meta
from champs_pipeline.data_prep.funnel import build_case_attribution, longest_text_per_case
from champs_pipeline.data_prep.llm_aggregate import (
    RECORD_LISTS, aggregate_findings, copy_concurred_site_records, normalised,
)
from champs_pipeline.data_prep.morphology_labels import load_schema
from champs_pipeline.data_prep.reports import extract_sections

SECTION_OF_SOURCE = {
    "site_report": "SITE_REPORT",
    "cpl_review_of_scans": "CPL_REVIEW",
    "cpl_slides": "CPL_SLIDES",
}


def check_run_meta(per_case_dir, schema, prompt_path):
    """Refuse a schema or prompt whose fingerprint differs from the run record."""
    run_meta = load_run_meta(per_case_dir.parent)
    if run_meta is None:
        raise SystemExit(f"no run_meta.json next to {per_case_dir}")
    problems = []
    current = {"schema": schema.schema_sha, "prompt": file_sha(prompt_path)}
    for name, sha in current.items():
        recorded = run_meta.get(f"{name}_sha")
        if sha != recorded:
            problems.append(f"{name} {sha} differs from the run's {recorded}")
    if problems:
        raise SystemExit("finalise with the schema and prompt the extraction used:\n  "
                         + "\n  ".join(problems))
    return run_meta


def load_records(per_case_dir):
    """``{case_id: payload}`` from the per-case files, with two repairs.

    A negation the model placed among the findings is moved to the
    negations; a finding without a name is dropped. Both are counted.
    """
    payloads = {}
    counts = {"files": 0, "moved_negations": 0, "dropped_malformed": 0}
    for path in sorted(per_case_dir.glob("*.json")):
        counts["files"] += 1
        record = json.loads(path.read_text())
        findings, extra_negations = [], []
        for entry in record.get("findings", []):
            if "condition" in entry:
                findings.append(entry)
            elif "negation_group" in entry:
                extra_negations.append(entry)
                counts["moved_negations"] += 1
            else:
                counts["dropped_malformed"] += 1
        payloads[path.stem] = {
            "findings": findings,
            "negations": record.get("negations", []) + extra_negations,
            "quality_flags": record.get("quality_flags", []),
            "uncertain": record.get("uncertain", []),
            "diagnoses": record.get("diagnoses", []),
        }
    return payloads, counts


def case_status(per_case_dir, payloads, corpus_path, findings):
    """One row per case: extracted, failed, or missing; a missing corpus case is an error."""
    failed = {p.name[: -len(".error.txt")] for p in per_case_dir.glob("*.error.txt")}
    failed -= set(payloads)  # a retried case leaves a stale error file
    corpus = set(payloads) | failed
    if corpus_path is not None:
        lines = Path(corpus_path).read_text().splitlines()
        corpus = {line.strip() for line in lines if line.strip()}
    rows_per_case = findings.groupby("champs_deid").size()
    rows = []
    for case_id in sorted(corpus | set(payloads) | failed):
        if case_id in payloads:
            status = "extracted"
        elif case_id in failed:
            status = "failed"
        else:
            status = "missing"
        rows.append({
            "champs_deid": case_id,
            "in_corpus": case_id in corpus,
            "status": status,
            "n_rows": int(rows_per_case.get(case_id, 0)),
        })
    table = pd.DataFrame(rows)
    missing = table[(table["status"] == "missing") & table["in_corpus"]]
    if len(missing):
        raise SystemExit(f"{len(missing)} corpus cases have neither a record nor an error file, "
                         f"for example {missing['champs_deid'].iloc[0]}")
    return table


def quote_checks(payloads, texts):
    """Share of quotes found in the report text, and per text source inside the named section.

    A quote is found when it occurs in the text with case and whitespace
    ignored. For the three headed sources the section is the one the
    model named; for ``implicit_cpl`` it is the whole text of a report
    without headers. Records copied for a concurrence are not counted.
    """
    found = total = 0
    per_source = {}
    for case_id, payload in payloads.items():
        text = texts.get(case_id, "")
        whole = normalised(text)
        sections = {k: normalised(v) for k, v in extract_sections(text).items() if v}
        for list_name, _, _ in RECORD_LISTS:
            for record in payload.get(list_name, []):
                if record.get("derived_from_concurrence"):
                    continue
                quote = normalised(record.get("supporting_quote"))
                total += 1
                found += bool(quote) and quote in whole
                source = record.get("text_source")
                if source == "implicit_cpl":
                    section = whole if not sections else ""
                else:
                    section = sections.get(SECTION_OF_SOURCE.get(source, ""), "")
                tally = per_source.setdefault(source, {"n": 0, "in_named_section": 0})
                tally["n"] += 1
                tally["in_named_section"] += bool(quote) and quote in section
    rate = round(found / total, 4) if total else None
    for tally in per_source.values():
        tally["rate"] = round(tally["in_named_section"] / tally["n"], 4) if tally["n"] else None
    return rate, per_source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-case", type=Path, required=True)
    ap.add_argument("--reports-csv", required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="corpus_cases.txt; every case in it must have a record or an error file.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    schema = load_schema(args.schema)
    run_meta = check_run_meta(args.per_case, schema, args.prompt)
    payloads, counts = load_records(args.per_case)
    print(f"{counts['files']} records; moved negations {counts['moved_negations']}, "
          f"dropped malformed findings {counts['dropped_malformed']}")

    counts["concurrence_copies"] = sum(copy_concurred_site_records(p) for p in payloads.values())
    findings, corrections = aggregate_findings(payloads, schema=schema)
    unnamed = findings["semantic_group"].isna() | (findings["semantic_group"] == "")
    findings = findings[~unnamed].reset_index(drop=True)
    # Modifier values are strings; a boolean-typed modifier the model sometimes
    # answers with a string would otherwise break the parquet type inference.
    for column in [c for c in findings.columns if c.startswith("mod_")]:
        findings[column] = findings[column].astype("string")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    findings.to_parquet(args.out_dir / "findings.parquet", index=False)
    print(f"findings.parquet: {len(findings)} rows, {findings['champs_deid'].nunique()} cases")

    status = case_status(args.per_case, payloads, args.corpus, findings)
    status.to_csv(args.out_dir / "case_status.csv", index=False)
    print(f"case_status.csv: {status['status'].value_counts().to_dict()}")

    reports = pd.read_csv(args.reports_csv)
    _, cases = build_case_attribution(reports)
    cases.to_parquet(args.out_dir / "case_attribution.parquet", index=False)
    print(f"case_attribution.parquet: {len(cases)} cases")

    verbatim, sections = quote_checks(payloads, longest_text_per_case(reports))
    record = {
        "model": run_meta.get("model"),
        "schema_version": schema.schema_version,
        "schema_sha": schema.schema_sha,
        "prompt_sha": file_sha(args.prompt),
        "settings": run_meta.get("settings", {}),
        "n_cases": len(payloads),
        "n_cases_with_records": int(findings["champs_deid"].nunique()),
        "n_failed": int((status["status"] == "failed").sum()),
        "n_rows": int(len(findings)),
        "unnamed_rows_dropped": int(unnamed.sum()),
        **{k: v for k, v in counts.items() if k != "files"},
        "verbatim_quote_rate": verbatim,
        "quote_in_named_section": sections,
        "corrections": corrections,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.out_dir / "extraction_record.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
