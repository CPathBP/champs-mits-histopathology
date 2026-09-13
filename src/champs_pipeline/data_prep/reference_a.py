"""Reference A: the report labels, per case unit and per slide.

The case reference states, per (case, organ, slide source, finding), what
the central laboratory's description of that organ asserts: ``positive``
when a finding record exists, ``uncertain`` when only a hedged record
exists, ``silent`` otherwise. The slide reference maps that onto the
training slides in three variants. ``raw`` keeps silence as a negative;
``elig`` masks a negative whose case asserts or hedges the finding
elsewhere, or whose description names a related finding or a quality
problem that makes silence uninformative; ``elig_screen`` also masks a
negative whose case a direct test screen names for that finding. A
positive is never masked.
"""

import re

import pandas as pd

from champs_pipeline.data_prep.cohort import UNIT, label_records

STATUSES = ("positive", "uncertain", "silent")
VARIANTS = ("raw", "elig", "elig_screen")
# Findings whose bare mention makes silence on another finding uninformative.
RELATED_FINDINGS = {
    "hyaline_membranes": ("diffuse_alveolar_damage", "fibrin_lining"),
    "hemozoin_pigment": ("pigment_unspecified",),
}
_FIBRIN_LINING = re.compile(r"\blin(?:ing|ed|es)\b", re.IGNORECASE)
_MODIFIER_PREFIX = "mod_"


def _severity_rank(schema):
    """Position of every severity value in its finding's ladder, for the higher-wins merge."""
    ranks = {}
    for condition in schema.conditions:
        ranks[condition.name] = {v: i for i, v in enumerate(condition.severity_values)}
    return ranks


def case_reference(findings, schema, organs):
    """Per (case, organ, slide source, finding): status, severity, extent, modifiers.

    Rows are the positive and uncertain findings of every described unit
    of the label sources; a finding without a row is silent in that unit.
    When a unit carries the finding in two text sources of the same slide
    source, the higher severity is kept.
    """
    rows = label_records(findings)
    rows = rows[rows["organ"].isin(organs) & rows["kind"].isin(["condition", "uncertain"])]
    ranks = _severity_rank(schema)
    modifiers = sorted(c for c in rows.columns if c.startswith(_MODIFIER_PREFIX))
    rows = rows.assign(
        status=rows["kind"].map({"condition": "positive", "uncertain": "uncertain"}),
        severity_rank=[ranks.get(f, {}).get(s, -1)
                       for f, s in zip(rows["semantic_group"], rows["severity"])],
    )
    # A positive beats an uncertain record; among positives the higher severity wins.
    rows = rows.sort_values(["status", "severity_rank"], ascending=[False, False])
    key = UNIT + ["semantic_group"]
    first = rows.drop_duplicates(key)
    flags = rows.groupby(key)[["derived_from_concurrence", "bilateral_replicated"]].any()
    reference = first.set_index(key)[["status", "severity", "extent"] + modifiers].join(flags)
    reference = reference.reset_index().rename(columns={"semantic_group": "finding"})
    return reference.sort_values(key[:-1] + ["finding"]).reset_index(drop=True)


def asserted_elsewhere(findings, finding, organs):
    """Cases with a positive or uncertain record of the finding in any target-organ section."""
    rows = findings[(findings["semantic_group"] == finding) & findings["organ"].isin(organs)
                    & findings["kind"].isin(["condition", "uncertain"])]
    return set(rows["champs_deid"])


def related_units(findings, related):
    """The units whose description names the related finding."""
    rows = label_records(findings)
    if related == "fibrin_lining":
        fibrin = rows[(rows["semantic_group"] == "fibrin") & (rows["kind"] == "condition")]
        quotes = fibrin["finding_examples"].fillna("")
        rows = fibrin[quotes.map(lambda q: bool(_FIBRIN_LINING.search(q)))]
    else:
        rows = rows[(rows["semantic_group"] == related) & (rows["kind"] == "condition")]
    return set(map(tuple, rows[UNIT].values))


def quality_units(findings):
    """The units whose description is out of focus or severely autolysed."""
    rows = label_records(findings)
    rows = rows[rows["kind"] == "quality"]
    blurred = rows["semantic_group"] == "out_of_focus"
    autolysed = (rows["semantic_group"] == "autolysis") & (rows["severity"] == "severe")
    return set(map(tuple, rows.loc[blurred | autolysed, UNIT].values))


def slide_reference(slides, reference, findings, screen, labels, organs):
    """Per (slide, finding, variant): the label, the keep mask and the reason for a mask.

    ``slides`` are the training slides of one organ group with their
    unit columns; ``reference`` is the case reference; ``screen`` lists
    (case_id, label) pairs whose negatives are uncertain.
    """
    status = reference.set_index(UNIT + ["finding"])
    screened = {(str(c), l) for c, l in zip(screen["case_id"], screen["label"])}
    quality = quality_units(findings)
    rows = []
    for finding in labels:
        elsewhere = asserted_elsewhere(findings, finding, organs)
        related = set()
        for name in RELATED_FINDINGS.get(finding, ()):
            related |= related_units(findings, name)
        for slide in slides.itertuples(index=False):
            unit = (slide.champs_deid, slide.organ, slide.slide_source)
            record = status.loc[unit + (finding,)] if unit + (finding,) in status.index else None
            for variant in VARIANTS:
                rows.append(_label_row(slide, finding, variant, record, unit, elsewhere,
                                       screened, related, quality))
    return pd.DataFrame(rows)


def _label_row(slide, finding, variant, record, unit, elsewhere, screened, related, quality):
    """One row of the slide reference."""
    row = {"slide_id": slide.slide_id, "champs_deid": slide.champs_deid, "organ": slide.organ,
           "slide_source": slide.slide_source, "finding": finding, "variant": variant}
    reason = None
    label = 0.0
    if record is not None and record["status"] == "positive":
        label = 1.0
    elif record is not None:
        reason = "uncertain"
    elif variant != "raw":
        screen = screened if variant == "elig_screen" else set()
        reason = _mask_reason(slide.champs_deid, finding, unit, elsewhere, screen, related,
                             quality)
    row["label"] = float("nan") if reason else label
    row["mask"] = 0.0 if reason else 1.0
    row["mask_reason"] = reason
    if record is not None:
        row.update({k: v for k, v in record.items() if k != "status"})
    return row


def _mask_reason(case_id, finding, unit, elsewhere, screened, related, quality):
    """Why a silent unit is not a negative in the eligibility variant, or None."""
    if case_id in elsewhere:
        return "eligibility"
    if (case_id, finding) in screened:
        return "screen"
    if unit in related:
        return "related_finding"
    if unit in quality:
        return "quality"
    return None


def qualifier_class(label, value, ladder):
    """The class index of a qualifier for an ordinal head: 0 absent, 1.. the ladder, -100 masked."""
    if pd.isna(label):
        return -100
    if label == 0:
        return 0
    if value is None or pd.isna(value) or value not in ladder:
        return -100
    return 1 + list(ladder).index(value)
