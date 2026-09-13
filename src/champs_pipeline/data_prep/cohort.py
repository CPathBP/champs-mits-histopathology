"""The slide cohort: which slides of the store enter the study, and why the others leave.

The cohort is built once, as an artifact, from the slide inventory, the
feature index, the stain scores, the findings table and the case
attribution. Every drop is one row of the drop log with its stage and
reason, so the funnel of the paper is a count over that log.

Two levels of inclusion are recorded per slide. A slide is *linked* when
its case has a report and some report section describes that organ for
that slide source; the linked cohort is the paper's cohort figure. A
slide is *training* when, in addition, the section that describes it is
authored by the central laboratory, it is not a quality-only or
inadequate description, the stain classifier calls it H&E, and a
correctly scaled feature file exists for the cohort encoder.
"""

import re

import pandas as pd

from champs_pipeline.data_prep.feature_index import usable_files
from champs_pipeline.data_prep.identifiers import study_id_in_name
from champs_pipeline.data_prep.reports import TEXT_SOURCE_TO_SLIDE_SOURCE, classify_slide_source
from champs_pipeline.data_prep.stains import is_he_slide
from champs_pipeline.data_prep.tissue import organ_of_slide, tissue_code

TARGET_ORGANS = ("right_lung", "left_lung", "liver")
ORGAN_GROUP = {"right_lung": "lung", "left_lung": "lung", "liver": "liver"}
# A section whose only records are these flags examined nothing.
NOT_EXAMINED = ("not_performed", "no_scans_available")
# The text sources that label a slide, per slide source. The site report
# describes site slides but is not a label source (the central laboratory's
# review of the same scans is).
LABEL_SOURCES = {
    "SITE": ("cpl_review_of_scans",),
    "CPL": ("cpl_slides", "implicit_cpl"),
}
UNIT = ["champs_deid", "organ", "slide_source"]
# A slide scanned again carries the scan time in its name; the newest scan stands for it.
_SCAN_TIME = re.compile(r"^(?P<stem>.+?) - (?P<time>\d{4}-\d\d-\d\d \d\d\.\d\d\.\d\d)$")

SLIDE_COLUMNS = [
    "slide_id", "champs_deid", "study_id", "site", "organ", "organ_group", "slide_source",
    "tissue_code", "scanner_power", "wsi_path", "n_files", "p_non_HE", "linked", "training",
    "drop_stage",
]


def slide_table(inventory, mapping):
    """One row per slide of the store with its case, organ and slide source.

    ``inventory`` is the crawl of the store (one row per file);
    ``mapping`` has ``study_id`` and ``champs_deid``. A slide id that
    names several files (a scan filed under two case directories, or two
    scans of one slide) keeps the file whose case directory matches the
    study id in its name, else the first by path; ``n_files`` records
    the count.
    """
    files = inventory.copy()
    files["slide_id"] = files["slide_id"].astype(str)
    named_case = files["slide_id"].map(study_id_in_name)
    files["_in_named_case"] = files["case_id"].astype(str) == named_case
    files = files.sort_values(["slide_id", "_in_named_case", "wsi_path"],
                              ascending=[True, False, True])
    slides = files.drop_duplicates("slide_id").drop(columns="_in_named_case").copy()
    slides["n_files"] = slides["slide_id"].map(files["slide_id"].value_counts())
    slides = slides.rename(columns={"case_id": "study_id"})
    deid = dict(zip(mapping["study_id"].astype(str), mapping["champs_deid"].astype(str)))
    slides["champs_deid"] = slides["study_id"].map(deid)
    slides["he_by_name"] = slides["slide_id"].map(is_he_slide)
    slides["tissue_code"] = slides["slide_id"].map(tissue_code)
    slides["organ"] = slides["slide_id"].map(organ_of_slide)
    slides["organ_group"] = slides["organ"].map(ORGAN_GROUP)
    file_names = slides["wsi_path"].astype(str).str.rsplit("/", n=1).str[-1]
    slides["slide_source"] = file_names.map(classify_slide_source)
    return slides


def content_records(findings):
    """The records that describe an organ: everything but the not-examined flags."""
    not_examined = (findings["kind"] == "quality") & findings["semantic_group"].isin(NOT_EXAMINED)
    return findings[~not_examined]


def described_units(findings):
    """The (case, organ, slide source) units some report section describes, by any source."""
    rows = content_records(findings)
    units = rows[["champs_deid", "organ", "text_source"]].drop_duplicates()
    units["slide_source"] = units["text_source"].map(TEXT_SOURCE_TO_SLIDE_SOURCE)
    return set(map(tuple, units[UNIT].values))


def label_records(findings):
    """The content records from the text sources that label a slide."""
    rows = content_records(findings)
    keep = pd.Series(False, index=rows.index)
    for slide_source, text_sources in LABEL_SOURCES.items():
        keep |= (rows["slide_source"] == slide_source) & rows["text_source"].isin(text_sources)
    return rows[keep]


def unit_flags(findings):
    """Per unit of the label sources: has a condition or negation, has the inadequate flag."""
    rows = label_records(findings)
    supervisable = rows["kind"].isin(["condition", "negation"])
    inadequate = (rows["kind"] == "quality") & (rows["semantic_group"] == "inadequate_for_dx")
    flags = pd.DataFrame({
        "supervisable": supervisable, "inadequate": inadequate,
        **{k: rows[k] for k in UNIT},
    })
    return flags.groupby(UNIT)[["supervisable", "inadequate"]].any()


def older_scans(slides):
    """The slides that are an earlier scan of a slide scanned again (same case and name stem)."""
    parts = slides["slide_id"].str.extract(_SCAN_TIME)
    scans = slides[parts["stem"].notna()].assign(stem=parts["stem"], time=parts["time"])
    newest = scans.sort_values("time").drop_duplicates(["champs_deid", "stem"], keep="last")
    return set(scans["slide_id"]) - set(newest["slide_id"])


class Funnel:
    """Applies the drop stages in order and keeps the log."""

    def __init__(self, slides):
        self.slides = slides
        self.dropped = []
        self.boxes = []

    def drop(self, stage, keep, reason):
        """Drop the slides where ``keep`` is False; record them and the box after the drop."""
        lost = self.slides[~keep]
        for slide_id in lost["slide_id"]:
            self.dropped.append({"stage": stage, "slide_id": slide_id, "reason": reason})
        self.slides = self.slides[keep]
        self.box(stage, reason)

    def box(self, stage, reason=""):
        """Record the count of the slides that remain, per organ, and the cases they belong to."""
        remaining = self.slides
        self.boxes.append({
            "stage": stage, "n": int(len(remaining)),
            "n_dropped": int(self.boxes[-1]["n"] - len(remaining)) if self.boxes else 0,
            "drop_reason": reason,
            "n_lung": int((remaining["organ_group"] == "lung").sum()),
            "n_liver": int((remaining["organ_group"] == "liver").sum()),
            "n_cases": int(remaining["champs_deid"].nunique()),
        })


def build_cohort(slides, index, stain_scores, findings, encoders, cohort_encoder):
    """The cohort tables from the stage inputs.

    Returns ``(cohort_slides, dropped, boxes)``: the slides of the target
    organs with their inclusion flags and per-encoder feature paths, the
    drop log, and the funnel boxes. ``encoders`` maps an encoder name to
    the resolution of its feature files.
    """
    content_cases = set(content_records(findings)["champs_deid"])
    units = described_units(findings)
    flags = unit_flags(findings)
    non_he = set(stain_scores.loc[stain_scores["p_non_HE"] >= 0.5, "slide_id"].astype(str))
    conflicts = set(index.loc[index["id_conflict"].isin([True, "True"]), "slide_id"])

    funnel = Funnel(slides)
    funnel.box("slides in the store")
    funnel.drop("not_he_by_name", funnel.slides["he_by_name"],
                "stain token in the file name")
    funnel.drop("no_tissue_code", funnel.slides["tissue_code"].notna(),
                "no MITS tissue code in the file name")
    funnel.drop("not_target_organ", funnel.slides["organ"].isin(TARGET_ORGANS),
                "tissue code of another organ")
    candidates = funnel.slides.copy()
    funnel.drop("no_case_mapping", funnel.slides["champs_deid"].notna(),
                "study id without a case id in the mapping")
    funnel.drop("case_id_conflict", ~funnel.slides["slide_id"].isin(conflicts),
                "study id in the file name differs from the case directory")
    funnel.drop("case_not_in_corpus", funnel.slides["champs_deid"].isin(content_cases),
                "case without an examined report")
    unit_keys = list(map(tuple, funnel.slides[UNIT].values))
    funnel.drop("no_examined_section_for_source",
                pd.Series([u in units for u in unit_keys], index=funnel.slides.index),
                "no report section describes this organ for this slide source")
    linked = set(funnel.slides["slide_id"])

    funnel.drop("not_he_by_classifier", ~funnel.slides["slide_id"].isin(non_he),
                "stain classifier probability of a non-H&E stain at or above 0.5")
    unit_flag = flags.reindex(pd.MultiIndex.from_frame(funnel.slides[UNIT]))
    funnel.drop("no_cpl_authored_record", unit_flag.index.isin(flags.index),
                "no record from the central laboratory's own description")
    unit_flag = flags.reindex(pd.MultiIndex.from_frame(funnel.slides[UNIT]))
    funnel.drop("quality_flags_only", unit_flag["supervisable"].to_numpy(),
                "the description carries quality flags only")
    unit_flag = flags.reindex(pd.MultiIndex.from_frame(funnel.slides[UNIT]))
    funnel.drop("inadequate_for_dx", ~unit_flag["inadequate"].to_numpy(),
                "the description flags the sample as inadequate for diagnosis")
    usable = usable_files(index, cohort_encoder, encoders[cohort_encoder])
    funnel.drop("no_usable_feature_file", funnel.slides["slide_id"].isin(set(usable["slide_id"])),
                f"no correctly scaled {cohort_encoder} feature file")
    funnel.drop("older_scan", ~funnel.slides["slide_id"].isin(older_scans(funnel.slides)),
                "an earlier scan of a slide that was scanned again")
    training = set(funnel.slides["slide_id"])

    cohort = candidates
    cohort["linked"] = cohort["slide_id"].isin(linked)
    cohort["training"] = cohort["slide_id"].isin(training)
    stage_of = {row["slide_id"]: row["stage"] for row in funnel.dropped}
    cohort["drop_stage"] = cohort["slide_id"].map(stage_of)
    scores = stain_scores[["slide_id", "p_non_HE"]].drop_duplicates("slide_id")
    cohort = cohort.merge(scores, on="slide_id", how="left")
    cohort = cohort[SLIDE_COLUMNS]
    for encoder, resolution in encoders.items():
        files = usable_files(index, encoder, resolution).set_index("slide_id")
        cohort[f"feature_path_{encoder}"] = cohort["slide_id"].map(files["path"])
        cohort[f"n_patches_{encoder}"] = cohort["slide_id"].map(files["n_patches"]).astype("Int64")
    dropped = pd.DataFrame(funnel.dropped, columns=["stage", "slide_id", "reason"])
    return cohort.reset_index(drop=True), dropped, funnel.boxes


def report_boxes(findings, attribution):
    """The report arm of the funnel: cases in the release, with text, with records, examined."""
    n_release = len(attribution)
    n_text = int(attribution["has_hetext"].sum())
    n_records = int(findings["champs_deid"].nunique())
    n_content = int(content_records(findings)["champs_deid"].nunique())
    return [
        {"stage": "cases in the release", "n": n_release, "n_dropped": 0, "drop_reason": ""},
        {"stage": "with report text", "n": n_text, "n_dropped": n_release - n_text,
         "drop_reason": "no text in the report"},
        {"stage": "with extracted records", "n": n_records, "n_dropped": n_text - n_records,
         "drop_reason": "the text yields no record"},
        {"stage": "with an examined section", "n": n_content, "n_dropped": n_records - n_content,
         "drop_reason": "only not-performed or no-scans flags"},
    ]


def check_cohort(cohort):
    """Refuse a cohort table that is not one row per slide with the expected flags."""
    missing = [c for c in SLIDE_COLUMNS if c not in cohort.columns]
    if missing:
        raise ValueError(f"cohort table lacks columns {missing}")
    if cohort["slide_id"].duplicated().any():
        raise ValueError("cohort table lists a slide twice")
    if (cohort["training"] & ~cohort["linked"]).any():
        raise ValueError("a training slide is not linked")
    if cohort.loc[cohort["training"], "champs_deid"].isna().any():
        raise ValueError("a training slide has no case id")
    return cohort
