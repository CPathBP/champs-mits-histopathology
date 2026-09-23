"""Render the supplementary table of the study cohort per site.

Reads the case and slide tables of the cohort and ``reference_a_slide.parquet``
and writes the display ``supplementary_table_sites`` as LaTeX and CSV with
a provenance record. For each of the seven sites the table gives the cases and slides of
the study cohort, the share of slides scanned at the site, and the
prevalence of each study finding with the corrected labels. The site of a
slide is the site under which the image archive holds it, which is known
for every slide; a case counts at the site of its slides.
"""

import argparse
from pathlib import Path

import pandas as pd

from champs_pipeline.figures.labels import FINDINGS, ORGANS, SITES
from champs_pipeline.figures.output import write_table

CORRECTED = "elig"
SITE_SCANNED = "SITE"


def site_name(archive_site):
    """The display name of a site from the name of its archive directory."""
    return archive_site.replace("_", " ")


def row(characteristic, level, cells):
    return {"Characteristic": characteristic, "Level": level, **cells}


def count_share(count, total):
    return f"{int(count):,} ({count / total:.1%})"


def study_slides(cohort_slides):
    """The study-cohort slides with the display name of their site."""
    slides = cohort_slides[cohort_slides["training"]].copy()
    slides["site_name"] = slides["site"].map(site_name)
    unlisted = set(slides["site_name"]) - set(SITES.values())
    if unlisted:
        raise ValueError(f"study-cohort slides at sites without a label: {unlisted}")
    if slides.groupby("champs_deid")["site_name"].nunique().max() > 1:
        raise ValueError("a case has slides at more than one site")
    return slides


def composition_rows(slides):
    """Cases, slides by organ, and the share of slides scanned at the site."""
    rows = []
    cases = {}
    for name in SITES.values():
        cases[name] = f"{slides.loc[slides['site_name'] == name, 'champs_deid'].nunique():,}"
    rows.append(row("Cases", "All", cases))
    for organ, label in ORGANS.items():
        cells = {}
        for name in SITES.values():
            selected = (slides["site_name"] == name) & (slides["organ_group"] == organ)
            cells[name] = f"{int(selected.sum()):,}"
        rows.append(row("Slides", label, cells))
    scanned = {}
    for name in SITES.values():
        site_slides = slides[slides["site_name"] == name]
        scanned[name] = count_share((site_slides["slide_source"] == SITE_SCANNED).sum(),
                                    len(site_slides))
    rows.append(row("Slides", "Scanned at the site", scanned))
    return rows


def prevalence_rows(slides, reference):
    """Positive slides and prevalence per finding and site, with the corrected labels."""
    site_of_slide = slides.set_index("slide_id")["site_name"]
    corrected = reference[(reference["variant"] == CORRECTED) & reference["label"].notna()].copy()
    corrected["site_name"] = corrected["slide_id"].map(site_of_slide)
    if corrected["site_name"].isna().any():
        raise ValueError("Reference A slides outside the study cohort")
    rows = []
    for organ, findings in FINDINGS.items():
        for finding, label in findings.items():
            selected = corrected[(corrected["organ_group"] == organ)
                                 & (corrected["finding"] == finding)]
            cells = {}
            for name in SITES.values():
                at_site = selected.loc[selected["site_name"] == name, "label"]
                cells[name] = count_share((at_site == 1).sum(), len(at_site))
            rows.append(row(f"Report-positive slides, {organ}", label, cells))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--reference-slides", type=Path, required=True,
                    help="reference_a/reference_a_slide.parquet")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    slides_path = args.cohort_dir / "cohort_slides.csv"
    cohort_slides = pd.read_csv(slides_path, low_memory=False,
                                usecols=["slide_id", "champs_deid", "site", "organ_group",
                                         "slide_source", "training"])
    reference = pd.read_parquet(args.reference_slides,
                                columns=["slide_id", "organ_group", "finding", "variant", "label"])
    slides = study_slides(cohort_slides)

    rows = composition_rows(slides) + prevalence_rows(slides, reference)
    table = pd.DataFrame(rows, columns=["Characteristic", "Level", *SITES.values()])
    write_table(table, args.out_dir, "supplementary_table_sites",
                [slides_path, args.reference_slides],
                align="ll" + "r" * len(SITES), group_column="Characteristic", header_width=12)


if __name__ == "__main__":
    main()
