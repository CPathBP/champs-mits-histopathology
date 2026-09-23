"""Render the flow diagram of case and slide selection (supplementary figure).

Reads ``funnel.json`` of the cohort and writes the display
``supplementary_figure_cohort_flow`` as PDF and PNG with the source data and
a provenance record. The
report arm counts cases and the slide arm counts slides. The arms meet where
slides are linked to the report sections that describe them; the linked
cohort is then reduced to the study cohort.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from champs_pipeline.figures import flow, style
from champs_pipeline.figures.output import save_figure

# Height of the drawing canvas; the saved figure is cropped to the drawn content.
CANVAS_HEIGHT_MM = 220
GAP_MM = 5
# Vertical space between the arms and the line that joins them, which carries its label.
JOIN_GAP_MM = 7

REPORT_ARM_LEFT = 0
REPORT_ARM_WIDTH = 38
REPORT_EXCLUSION_LEFT = 42
REPORT_EXCLUSION_WIDTH = 42
SLIDE_ARM_LEFT = 88
SLIDE_ARM_WIDTH = 44
SLIDE_EXCLUSION_LEFT = 136
COHORT_FILL = style.tint(style.GREY, 0.22)
SLIDE_EXCLUSION_WIDTH = style.FULL_WIDTH_MM - SLIDE_EXCLUSION_LEFT
LINKAGE_LABEL = "Linked by case, organ and slide source"

# Exclusion reasons by the stage that the funnel names after the excluded items.
REPORT_EXCLUSIONS = {
    "with report text": "No report text",
    "with extracted records": "No record extracted from the report text",
    "with an examined section": "No examined organ section",
}
SLIDE_EXCLUSIONS = {
    "not_he_by_name": "Stain named in the file name",
    "no_tissue_code": "No tissue code in the file name",
    "not_target_organ": "Organ other than lung or liver",
    "no_case_mapping": "No case identifier",
    "case_id_conflict": "Conflicting case identifiers",
}
LINKAGE_EXCLUSIONS = {
    "case_not_in_corpus": "Case without an examined report",
    "no_examined_section_for_source": "No report section describing the organ and slide source",
}
STUDY_EXCLUSIONS = {
    "not_he_by_classifier": "Not H&E according to the stain classifier",
    "no_cpl_authored_record": "No description by the central laboratory",
    "inadequate_for_dx": "Inadequate for diagnosis",
    "quality_flags_only": "No finding or negation in the description",
    "no_usable_feature_file": "No feature file at the required magnification",
    "older_scan": "Earlier scan of a rescanned slide",
}


def check_stages(funnel):
    """Every exclusion stage of the funnel must have a label, in the order of the funnel."""
    report_stages = [step["stage"] for step in funnel["report_arm"][1:]]
    slide_stages = [step["stage"] for step in funnel["slide_arm"][1:]]
    if report_stages != list(REPORT_EXCLUSIONS):
        raise ValueError(f"report arm stages without matching labels: {report_stages}")
    labelled = [*SLIDE_EXCLUSIONS, *LINKAGE_EXCLUSIONS, *STUDY_EXCLUSIONS]
    if slide_stages != labelled:
        raise ValueError(f"slide arm stages without matching labels: {slide_stages}")


def steps(arm, stages):
    return [step for step in arm if step["stage"] in stages]


def exclusion_box(ax, left, top, width, heading, excluded, labels):
    """A box with the total of excluded items as its title and one line per reason."""
    total = sum(step["n_dropped"] for step in excluded)
    lines = []
    for step in excluded:
        lines.append(f"{flow.LIST_MARKER}{labels[step['stage']]} ({step['n_dropped']:,})")
    return flow.box(ax, left, top, width, "\n".join(lines), title=f"{heading} (n = {total:,})",
                    centered=False)


def slide_count_box(ax, left, top, width, heading, step):
    """A box with a cohort name as its title and its slide and case counts."""
    return flow.box(ax, left, top, width,
                    f"{step['n']:,} slides ({step['n_lung']:,} lung, {step['n_liver']:,} liver) "
                    f"from {step['n_cases']:,} cases", title=heading, facecolor=COHORT_FILL)


def source_data(funnel):
    rows = []
    for arm in ("report_arm", "slide_arm"):
        for step in funnel[arm]:
            rows.append({"arm": arm, **step})
    return pd.DataFrame(rows)


def draw(funnel):
    report = funnel["report_arm"]
    slides = funnel["slide_arm"]
    last_before_linkage = steps(slides, SLIDE_EXCLUSIONS)[-1]
    linked = steps(slides, LINKAGE_EXCLUSIONS)[-1]
    study = steps(slides, STUDY_EXCLUSIONS)[-1]

    fig, ax = flow.canvas(style.FULL_WIDTH_MM, CANVAS_HEIGHT_MM)
    top = CANVAS_HEIGHT_MM - 1
    report_start = flow.box(ax, REPORT_ARM_LEFT, top, REPORT_ARM_WIDTH,
                            f"n = {report[0]['n']:,}", title="Cases in the release")
    slide_start = flow.box(ax, SLIDE_ARM_LEFT, top, SLIDE_ARM_WIDTH,
                           f"n = {slides[0]['n']:,}", title="Slides in the image archive")

    exclusion_top = min(report_start.bottom, slide_start.bottom) - GAP_MM
    report_excluded = exclusion_box(ax, REPORT_EXCLUSION_LEFT, exclusion_top,
                                    REPORT_EXCLUSION_WIDTH, "Excluded cases", report[1:],
                                    REPORT_EXCLUSIONS)
    slide_excluded = exclusion_box(ax, SLIDE_EXCLUSION_LEFT, exclusion_top, SLIDE_EXCLUSION_WIDTH,
                                   "Excluded slides", steps(slides, SLIDE_EXCLUSIONS),
                                   SLIDE_EXCLUSIONS)

    second_top = min(report_excluded.bottom, slide_excluded.bottom) - GAP_MM
    report_examined = flow.box(ax, REPORT_ARM_LEFT, second_top, REPORT_ARM_WIDTH,
                               f"n = {report[-1]['n']:,}", title="Cases with an examined report")
    slide_candidates = flow.box(ax, SLIDE_ARM_LEFT, second_top, SLIDE_ARM_WIDTH,
                                f"n = {last_before_linkage['n']:,}",
                                title="Lung and liver H&E slides with a case identifier")

    join_y = min(report_examined.bottom, slide_candidates.bottom) - JOIN_GAP_MM
    linkage_top = join_y - GAP_MM / 2
    not_linked = exclusion_box(ax, SLIDE_EXCLUSION_LEFT, linkage_top, SLIDE_EXCLUSION_WIDTH,
                               "Not linked", steps(slides, LINKAGE_EXCLUSIONS), LINKAGE_EXCLUSIONS)
    linked_box = slide_count_box(ax, SLIDE_ARM_LEFT, not_linked.bottom - GAP_MM, SLIDE_ARM_WIDTH,
                                 "Linked cohort", linked)

    study_excluded = exclusion_box(ax, SLIDE_EXCLUSION_LEFT, linked_box.bottom - GAP_MM,
                                   SLIDE_EXCLUSION_WIDTH, "Excluded slides",
                                   steps(slides, STUDY_EXCLUSIONS), STUDY_EXCLUSIONS)
    study_box = slide_count_box(ax, SLIDE_ARM_LEFT, study_excluded.bottom - GAP_MM,
                                SLIDE_ARM_WIDTH, "Study cohort", study)

    report_x = report_start.center_x
    slide_x = slide_start.center_x
    flow.arrow(ax, (report_x, report_start.bottom), (report_x, report_examined.top))
    flow.arrow(ax, (report_x, report_excluded.middle_y), (report_excluded.left,
                                                          report_excluded.middle_y))
    flow.arrow(ax, (slide_x, slide_start.bottom), (slide_x, slide_candidates.top))
    flow.arrow(ax, (slide_x, slide_excluded.middle_y), (slide_excluded.left,
                                                        slide_excluded.middle_y))
    flow.line(ax, [(report_x, report_examined.bottom), (report_x, join_y)])
    flow.arrow(ax, (report_x, join_y), (slide_x, join_y))
    ax.text((report_x + slide_x) / 2, join_y + flow.PADDING_MM / 2, LINKAGE_LABEL,
            ha="center", va="bottom")
    flow.arrow(ax, (slide_x, slide_candidates.bottom), (slide_x, linked_box.top))
    flow.arrow(ax, (slide_x, not_linked.middle_y), (not_linked.left, not_linked.middle_y))
    flow.arrow(ax, (slide_x, linked_box.bottom), (slide_x, study_box.top))
    flow.arrow(ax, (slide_x, study_excluded.middle_y), (study_excluded.left,
                                                        study_excluded.middle_y))
    flow.crop(fig, ax, study_box.bottom)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--funnel", type=Path, required=True, help="cohort/funnel.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    funnel = json.loads(args.funnel.read_text())
    check_stages(funnel)
    style.apply()
    fig = draw(funnel)
    save_figure(fig, args.out_dir, "supplementary_figure_cohort_flow",
                {"": source_data(funnel)}, [args.funnel])


if __name__ == "__main__":
    main()
