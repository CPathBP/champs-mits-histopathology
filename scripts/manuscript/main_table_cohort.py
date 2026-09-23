"""Render the cohort table and its full version: characteristics of the cases and cohorts.

Reads the case and slide tables of the cohort and the finding supply table,
and writes two displays as LaTeX and CSV with a provenance record.
``supplementary_table_cohort`` gives the examined cases, the linked cohort
and the study cohort, with the slide counts and per-site breakdown.
``main_table_cohort`` gives the examined cases and the study cohort,
with cohort characteristics and Reference A-positive cases arranged in two
side-by-side panels; it omits only the linked cohort and slide counts.
Shares of demographic characteristics are computed among the cases with a
demographics record. Shares of Reference A positive cases are computed among
the study-cohort cases with a slide of the organ concerned.
"""

import argparse
from pathlib import Path

import pandas as pd

from champs_pipeline.figures import latex
from champs_pipeline.figures.labels import FINDINGS, NOT_APPLICABLE, ORGANS, SITES, SLIDE_SOURCES
from champs_pipeline.figures.output import write_table

# Column header and the inclusion flag of the case and slide tables.
COHORTS = {"Examined cases": "examined", "Linked cohort": "linked", "Study cohort": "training"}
STUDY_COHORT = "Study cohort"
# The main table leaves out the linked cohort, which the full table in the supplement gives.
MAIN_COLUMNS = ["Examined cases", STUDY_COHORT]

DEATH_CATEGORIES = {"stillbirth": "Stillbirth", "neonate": "Neonate",
                    "infant/child": "Infant or child"}
SEXES = {"Male": "Male", "Female": "Female", "Indeterminate or Ambiguous": "Indeterminate"}
YEAR_EDGES = [2015, 2019, 2021, 2024]
YEAR_LABELS = ["2016–2019", "2020–2021", "2022–2024"]
# Characteristics reported in the table; the table note states that none has missing values.
REPORTED_COLUMNS = ["death_category", "sex", "site_iso_code", "death_year",
                    "calc_postmortem_hrs"]


def row(characteristic, level, cells):
    return {"Characteristic": characteristic, "Level": level, **cells}


def count_share(mask):
    """Count and percentage of the rows where ``mask`` holds, as ``1,234 (12.3%)``."""
    return f"{int(mask.sum()):,} ({mask.mean():.1%})"


def median_iqr(values):
    """Median and interquartile range, as ``10 (3–18)``."""
    first, median, third = values.quantile([0.25, 0.5, 0.75])
    return f"{median:.0f} ({first:.0f}–{third:.0f})"


def categorical_rows(groups, characteristic, column, levels):
    """One row per level; every value of the column must be one of the levels."""
    for name, frame in groups.items():
        unlisted = set(frame[column].dropna()) - set(levels)
        if unlisted:
            raise ValueError(f"{characteristic}: values without a label in {name}: {unlisted}")
    rows = []
    for value, label in levels.items():
        cells = {name: count_share(frame[column] == value) for name, frame in groups.items()}
        rows.append(row(characteristic, label, cells))
    return rows


def case_rows(cases):
    """Case counts and the demographic characteristics of the three case sets."""
    cases = cases.assign(year_of_death=pd.cut(cases["death_year"], YEAR_EDGES,
                                              labels=YEAR_LABELS))
    groups = {name: cases[cases[flag] & cases["in_demographics"]]
              for name, flag in COHORTS.items()}
    for name, frame in groups.items():
        missing = [column for column in REPORTED_COLUMNS if frame[column].isna().any()]
        if missing:
            raise ValueError(f"{name}: missing values in {missing}; the table note says none")

    rows = [
        row("Cases", "All", {name: f"{int(cases[flag].sum()):,}"
                             for name, flag in COHORTS.items()}),
        row("Cases", "With a demographics record",
            {name: f"{len(frame):,}" for name, frame in groups.items()}),
    ]
    rows += categorical_rows(groups, "Death category", "death_category", DEATH_CATEGORIES)
    rows += categorical_rows(groups, "Sex", "sex", SEXES)
    pmi = {name: median_iqr(frame["calc_postmortem_hrs"]) for name, frame in groups.items()}
    rows.append(row("Post-mortem interval, h", "Median (IQR)", pmi))
    rows += categorical_rows(groups, "Site", "site_iso_code", SITES)
    year_levels = {label: label for label in YEAR_LABELS}
    rows += categorical_rows(groups, "Year of death", "year_of_death", year_levels)
    return rows


def slide_rows(slides):
    """Slide counts by organ and by scanning location; slides exist from the linked cohort on."""
    rows = []
    for characteristic, column, levels in [("Slides, organ", "organ_group", ORGANS),
                                           ("Slides, scanned at", "slide_source", SLIDE_SOURCES)]:
        for value, label in levels.items():
            cells = {"Examined cases": NOT_APPLICABLE}
            for name, flag in COHORTS.items():
                if flag != "examined":
                    selected = slides[flag] & (slides[column] == value)
                    cells[name] = f"{int(selected.sum()):,}"
            rows.append(row(characteristic, label, cells))
    return rows


def reference_a_rows(slides, supply):
    """Study-cohort cases with a Reference A positive slide, per finding."""
    study_slides = slides[slides["training"]]
    supply = supply.set_index(["organ", "finding"])
    rows = []
    for organ, findings in FINDINGS.items():
        organ_cases = study_slides.loc[study_slides["organ_group"] == organ, "champs_deid"]
        denominator = organ_cases.nunique()
        for finding, label in findings.items():
            positive = int(supply.loc[(organ, finding), "cases_with_positive_training_slide"])
            cells = {name: NOT_APPLICABLE for name in COHORTS}
            cells[STUDY_COHORT] = f"{positive:,} ({positive / denominator:.1%})"
            rows.append(row(f"Report-positive cases, {organ}", label, cells))
    return rows


def main_table_tex(demographics, reference_a):
    """Render compact side-by-side panels for the main cohort table."""
    characteristics = pd.DataFrame(
        [item for item in demographics if item["Characteristic"] != "Cases"],
        columns=["Characteristic", "Level", *MAIN_COLUMNS],
    )
    left = latex.tabular(
        characteristics,
        align=(r"@{}>{\raggedright\arraybackslash}p{2.8cm}"
               r">{\raggedright\arraybackslash}p{2.4cm}rr@{}"),
        group_column="Characteristic",
        header_width=10,
    )

    right_lines = [
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.8cm}r@{}}",
        r"\toprule",
        "Finding & " + latex.header_cell("Study cohort cases, n (%)", "r", 12) + r" \\",
        r"\midrule",
    ]
    previous_organ = None
    for item in reference_a:
        organ = item["Characteristic"].rsplit(", ", maxsplit=1)[-1].capitalize()
        if organ != previous_organ:
            if previous_organ is not None:
                right_lines.append(r"\addlinespace")
            right_lines.append(
                rf"\multicolumn{{2}}{{@{{}}l}}{{\textit{{{latex.escape(organ)}}}}} \\"
            )
            previous_organ = organ
        right_lines.append(latex.row([item["Level"], item[STUDY_COHORT]]))
    right_lines += [r"\bottomrule", r"\end{tabular}"]
    right = "\n".join(right_lines) + "\n"

    return (
        r"\begin{minipage}[t]{0.67\linewidth}" "\n"
        r"\vspace{0pt}\textbf{a}\enspace Cohort characteristics\par\smallskip" "\n"
        r"\centering" "\n"
        f"{left}"
        r"\end{minipage}\hfill%" "\n"
        r"\begin{minipage}[t]{0.31\linewidth}" "\n"
        r"\vspace{0pt}\textbf{b}\enspace Report-positive cases\par\smallskip" "\n"
        r"\centering" "\n"
        f"{right}"
        r"\end{minipage}" "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--finding-supply", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cases_path = args.cohort_dir / "cohort_cases.csv"
    slides_path = args.cohort_dir / "cohort_slides.csv"
    cases = pd.read_csv(cases_path, low_memory=False)
    slides = pd.read_csv(slides_path, low_memory=False,
                         usecols=["champs_deid", "organ_group", "slide_source",
                                  "linked", "training"])
    supply = pd.read_csv(args.finding_supply)
    inputs = [cases_path, slides_path, args.finding_supply]

    demographics = case_rows(cases)
    reference_a = reference_a_rows(slides, supply)
    full = pd.DataFrame(demographics + slide_rows(slides) + reference_a,
                        columns=["Characteristic", "Level", *COHORTS])
    write_table(full, args.out_dir, "supplementary_table_cohort", inputs,
                align="llrrr", group_column="Characteristic", header_width=10)
    table = pd.DataFrame(demographics + reference_a,
                         columns=["Characteristic", "Level", *MAIN_COLUMNS])
    write_table(table, args.out_dir, "main_table_cohort", inputs,
                align="llrr", group_column="Characteristic", header_width=10)
    main_tex_path = args.out_dir / "main_table_cohort" / "main_table_cohort.tex"
    main_tex_path.write_text(main_table_tex(demographics, reference_a))


if __name__ == "__main__":
    main()
