"""Render the supplementary table of candidate findings and the selection of the study findings.

Reads ``candidate_table.csv`` and writes the display
``supplementary_table_candidate_findings`` as LaTeX and CSV with a provenance
record. The table lists
every finding with at least ten positive Reference A units in the lungs or
the liver, ordered by organ and by the number of positive cases. Positive
cases are given with their share of the examined cases; the chain share is
the share of examined cases with a DeCoDe record whose causal chain holds a
cause of death that the diagnosis standard ties to the finding.
"""

import argparse
from pathlib import Path

import pandas as pd

from champs_pipeline.figures.labels import NOT_APPLICABLE, ORGANS, finding_name
from champs_pipeline.figures.output import write_table

COLUMNS = {
    "organ": "Organ",
    "finding": "Finding",
    "positive_cases": "Positive cases, n (%)",
    "cause_conditions": "Causes of death supported by the finding",
    "cause_chain_share": "Causal chains, %",
    "selected": "Selected",
}


def yes_no(value):
    if value:
        return "Yes"
    return "No"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-table", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    candidates = pd.read_csv(args.candidate_table)
    organ_order = pd.Categorical(candidates["organ"], categories=list(ORGANS), ordered=True)
    candidates = (candidates.assign(organ_order=organ_order)
                  .sort_values(["organ_order", "positive_cases"], ascending=[True, False]))

    rows = []
    for record in candidates.itertuples(index=False):
        causes = record.cause_conditions
        if pd.isna(causes) or not causes:
            causes = NOT_APPLICABLE
        rows.append({
            COLUMNS["organ"]: ORGANS[record.organ],
            COLUMNS["finding"]: finding_name(record.finding, record.organ),
            COLUMNS["positive_cases"]: f"{record.positive_cases:,} ({record.case_prevalence:.1%})",
            COLUMNS["cause_conditions"]: causes,
            COLUMNS["cause_chain_share"]: f"{record.cause_chain_share:.2%}",
            COLUMNS["selected"]: yes_no(record.selected),
        })
    table = pd.DataFrame(rows, columns=list(COLUMNS.values()))
    write_table(table, args.out_dir, "supplementary_table_candidate_findings",
                [args.candidate_table],
                align=r"llr>{\raggedright\arraybackslash}p{4cm}rl",
                group_column=COLUMNS["organ"], header_width=10,
                long=True)


if __name__ == "__main__":
    main()
