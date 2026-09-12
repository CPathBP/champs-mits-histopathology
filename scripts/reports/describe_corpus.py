"""Describe the extracted corpus without a model.

Writes tables and a summary into ``--out-dir``:
``records_by_organ_source_kind.csv`` (rows and cases per organ, text source
and kind, with the rows copied for a concurrence counted apart),
``qualifiers_by_finding.csv`` (how often each severity, extent and
modifier value occurs per finding), ``cases_by_template_origin.csv``
(cases per reporting template and text origin),
``text_source_by_origin.csv`` (the model's text source against the
model-free origin of the case), ``coverage_by_organ_source.csv`` (per
organ and slide source, and per organ and text source: corpus cases with
any record, with a condition, with a negation, with a quality flag only,
and with nothing), and ``corpus_summary.json``.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def records_by_organ_source_kind(findings):
    grouped = findings.groupby(["organ", "text_source", "kind"])
    table = grouped.agg(
        n_rows=("champs_deid", "size"),
        n_cases=("champs_deid", "nunique"),
        n_rows_from_concurrence=("derived_from_concurrence", "sum"),
    )
    return table.reset_index()


def qualifiers_by_finding(findings):
    """Long table: finding, qualifier (severity, extent, or a modifier), value, count."""
    qualifiers = ["severity", "extent"] + [c for c in findings.columns if c.startswith("mod_")]
    parts = []
    for qualifier in qualifiers:
        counts = findings.groupby(["semantic_group", qualifier]).size().reset_index(name="n_rows")
        counts = counts.rename(columns={qualifier: "value"})
        counts.insert(1, "qualifier", qualifier.removeprefix("mod_"))
        parts.append(counts)
    table = pd.concat(parts, ignore_index=True)
    return table.sort_values(["semantic_group", "qualifier", "value"]).reset_index(drop=True)


def cases_by_template_origin(attribution):
    grouped = attribution.groupby(["template", "origin"])
    return grouped.agg(n_cases=("champs_deid", "size")).reset_index()


def text_source_by_origin(findings, attribution):
    """Rows and cases per (model-free origin of the case, text source the model named)."""
    merged = findings.merge(attribution[["champs_deid", "origin"]], on="champs_deid", how="left")
    grouped = merged.groupby(["origin", "text_source"])
    table = grouped.agg(n_rows=("champs_deid", "size"), n_cases=("champs_deid", "nunique"))
    return table.reset_index()


def coverage(findings, corpus, by):
    """Per (organ, ``by``): corpus cases with any record, a condition, a negation, or nothing."""
    rows = []
    units = findings[["organ", by]].drop_duplicates()
    for organ, source in units.itertuples(index=False):
        here = findings[(findings["organ"] == organ) & (findings[by] == source)]
        cases_with = {kind: set(here.loc[here["kind"] == kind, "champs_deid"]) for kind in
                      ("condition", "negation", "quality", "uncertain", "diagnosis")}
        any_record = set(here["champs_deid"])
        substantive = cases_with["condition"] | cases_with["negation"] | cases_with["uncertain"]
        only_flags = any_record - substantive
        rows.append({
            "organ": organ, by: source,
            "n_corpus_cases": len(corpus),
            "n_any_record": len(any_record & corpus),
            "n_condition": len(cases_with["condition"] & corpus),
            "n_negation": len(cases_with["negation"] & corpus),
            "n_uncertain": len(cases_with["uncertain"] & corpus),
            "n_quality_flag_only": len(only_flags & corpus),
            "n_nothing": len(corpus - any_record),
        })
    return pd.DataFrame(rows).sort_values(["organ", by]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--findings", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--corpus", required=True, help="corpus_cases.txt, the denominator.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    findings = pd.read_parquet(args.findings)
    attribution = pd.read_parquet(args.attribution)
    corpus = {line.strip() for line in Path(args.corpus).read_text().splitlines() if line.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_by_organ_source_kind(findings).to_csv(
        args.out_dir / "records_by_organ_source_kind.csv", index=False)
    qualifiers_by_finding(findings).to_csv(args.out_dir / "qualifiers_by_finding.csv", index=False)
    cases_by_template_origin(attribution).to_csv(
        args.out_dir / "cases_by_template_origin.csv", index=False)
    text_source_by_origin(findings, attribution).to_csv(
        args.out_dir / "text_source_by_origin.csv", index=False)
    by_slide = coverage(findings, corpus, "slide_source")
    by_text = coverage(findings, corpus, "text_source")
    pd.concat([by_slide.rename(columns={"slide_source": "source"}).assign(source_kind="slide"),
               by_text.rename(columns={"text_source": "source"}).assign(source_kind="text")],
              ignore_index=True).to_csv(args.out_dir / "coverage_by_organ_source.csv", index=False)

    summary = {
        "n_cases_in_report_table": int(len(attribution)),
        "n_cases_with_hetext": int(attribution["has_hetext"].sum()),
        "n_corpus_cases": len(corpus),
        "n_cases_with_records": int(findings["champs_deid"].nunique()),
        "n_rows": int(len(findings)),
        "n_rows_from_concurrence": int(findings["derived_from_concurrence"].sum()),
        "n_rows_bilateral_replicated": int(findings["bilateral_replicated"].sum()),
        "n_rows_by_kind": findings["kind"].value_counts().to_dict(),
    }
    (args.out_dir / "corpus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
