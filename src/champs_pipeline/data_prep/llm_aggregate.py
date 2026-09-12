"""Flatten the per-case extraction records into the findings table.

One row per (case, organ, text source, kind, finding). ``kind`` is
``condition``, ``negation``, ``quality``, ``uncertain`` or ``diagnosis``. A
row carries the severity and extent of the finding, one column per modifier
key seen in the corpus (``mod_<key>``), the supporting quotes joined with
`` | ``, whether the record was copied from the site report because the CPL
review concurred with it, and whether the same record exists for the paired
organ (the other lung, the other CNS sample).
"""

import re
import warnings

import pandas as pd

from champs_pipeline.data_prep.reports import TEXT_SOURCE_TO_SLIDE_SOURCE

MAX_QUOTES = 5
CONCURRENCE_FLAG = "concurs_with_site"
# Paired organs a combined header ("Lungs:", "CNS:") replicates a finding to.
PAIRED_ORGANS = {
    "right_lung": "left_lung", "left_lung": "right_lung",
    "cns_post": "cns_trans", "cns_trans": "cns_post",
}

# The first block is the identity of a row. `text_organ_key` repeats the organ
# and `source` repeats the slide source: the cohort construction reads them
# under these names.
BASE_COLUMNS = [
    "champs_deid", "organ", "slide_source", "text_source",
    "semantic_group", "finding_examples", "text_organ_key", "source",
    "severity", "extent", "kind", "derived_from_concurrence", "bilateral_replicated",
]
# (payload list, kind, key that names the finding)
RECORD_LISTS = (
    ("findings", "condition", "condition"),
    ("negations", "negation", "negation_group"),
    ("quality_flags", "quality", "flag"),
    ("uncertain", "uncertain", "condition"),
    ("diagnoses", "diagnosis", "diagnosis"),
)


def normalised(quote):
    """A quote with case and whitespace removed, for comparisons."""
    return re.sub(r"\s+", "", (quote or "").lower())


# A concurrence flag under a section copies the records of the section it endorses.
ENDORSED_SOURCE = {"cpl_review_of_scans": "site_report", "cpl_slides": "cpl_review_of_scans"}


def copy_concurred_site_records(payload):
    """Copy the endorsed section's records into the section that concurs with it.

    A ``concurs_with_site`` quality flag under ``cpl_review_of_scans`` means
    the CPL pathologist endorsed the site report for that organ; under
    ``cpl_slides`` it means an organ line only referred back to the CPL
    review. The endorsed section's findings, negations and uncertain
    entries for that organ are appended under the concurring section with
    ``derived_from_concurrence`` set, unless that section carries its own
    record of the same kind and name for the organ. The review is copied
    before the slides section, so a chain of endorsements carries through.
    Returns the number of copies.
    """
    copies = 0
    for target, endorsed_source in ENDORSED_SOURCE.items():
        endorsed = {
            q["organ"] for q in payload.get("quality_flags", [])
            if q.get("flag") == CONCURRENCE_FLAG and q.get("text_source") == target
        }
        if not endorsed:
            continue
        for list_name, _, name_key in RECORD_LISTS:
            if list_name in ("quality_flags", "diagnoses"):
                continue
            records = payload.get(list_name, [])
            own = {(r["organ"], r.get(name_key)) for r in records
                   if r.get("text_source") == target}
            for record in list(records):
                if record.get("text_source") != endorsed_source or record["organ"] not in endorsed:
                    continue
                if (record["organ"], record.get(name_key)) in own:
                    continue
                records.append(dict(record, text_source=target, derived_from_concurrence=True))
                copies += 1
    return copies


def _allowed(schema):
    """Per finding name: allowed organs, severities, extents, and modifier values."""
    allowed = {}
    if schema is None:
        return allowed
    entries = schema.conditions + schema.negation_flags + schema.quality_flags + schema.diagnoses
    for spec in entries:
        allowed[spec.name] = {
            "organs": set(spec.organs),
            "severity": set(getattr(spec, "severity_values", ())),
            "severity_order": list(getattr(spec, "severity_values", ())),
            "extent": set(getattr(spec, "extent_values", ())),
            "modifiers": {k: set(v) for k, v in getattr(spec, "modifiers", {}).items()},
        }
    return allowed


class Aggregation:
    """The rows of the findings table and the counts of what was corrected."""

    def __init__(self, schema):
        self.allowed = _allowed(schema)
        self.rows = {}
        self.modifier_keys = set()
        self.dropped = {}   # (reason, finding, value) -> count
        self.nulled = {}    # (finding, field, value) -> count
        self.merged = {}    # (event, finding) -> count: quotes beyond the cap, severity conflicts

    def _count(self, table, key):
        table[key] = table.get(key, 0) + 1

    def _validate(self, name, organ, severity, extent, modifiers):
        """Apply the finding's own organs, ladders and modifier values.

        Returns ``None`` when the organ is not one of the finding's organs.
        """
        rules = self.allowed.get(name)
        if rules is None:
            return severity, extent, modifiers
        if organ not in rules["organs"]:
            self._count(self.dropped, ("organ_not_allowed", name, organ))
            return None
        if severity is not None and severity not in rules["severity"]:
            self._count(self.nulled, (name, "severity", severity))
            severity = None
        if extent is not None and extent not in rules["extent"]:
            self._count(self.nulled, (name, "extent", extent))
            extent = None
        if "reason" in modifiers:  # an uncertain record; its reason is not a modifier
            return severity, extent, modifiers
        kept = {}
        for key, value in modifiers.items():
            if value is None:
                continue
            if key not in rules["modifiers"] or str(value) not in rules["modifiers"][key]:
                self._count(self.nulled, (name, key, str(value)))
                continue
            kept[key] = value
        return severity, extent, kept

    def ingest(self, case_id, record, kind, name_key):
        name = record.get(name_key)
        organ, text_source = record["organ"], record["text_source"]
        slide_source = TEXT_SOURCE_TO_SLIDE_SOURCE.get(text_source)
        if slide_source is None:
            self._count(self.dropped, ("unknown_text_source", name, str(text_source)))
            return
        modifiers = dict(record.get("modifiers") or {})
        if kind == "uncertain":
            modifiers = {"reason": record.get("reason", "hedge")}
        validated = self._validate(name, organ, record.get("severity"), record.get("extent"),
                                   modifiers)
        if validated is None:
            return
        severity, extent, modifiers = validated
        self.modifier_keys.update(modifiers)
        quote = record.get("supporting_quote", "")
        key = (case_id, organ, text_source, kind, name)
        row = self.rows.get(key)
        if row is None:
            self.rows[key] = {
                "champs_deid": case_id,
                "organ": organ,
                "slide_source": slide_source,
                "text_source": text_source,
                "kind": kind,
                "semantic_group": name,
                "severity": severity,
                "extent": extent,
                "_quotes": [quote] if quote else [],
                "_mods": modifiers,
                "text_organ_key": organ,
                "source": slide_source,
                "derived_from_concurrence": bool(record.get("derived_from_concurrence")),
                "bilateral_replicated": False,
            }
            return
        if quote and quote not in row["_quotes"]:
            if len(row["_quotes"]) < MAX_QUOTES:
                row["_quotes"].append(quote)
            else:
                self._count(self.merged, ("quotes_beyond_cap", name))
        row["severity"] = self._higher_severity(name, row["severity"], severity)
        if row["extent"] is None:
            row["extent"] = extent
        for key_, value in modifiers.items():
            if row["_mods"].get(key_) is None:
                row["_mods"][key_] = value

    def _higher_severity(self, name, current, new):
        """The higher rung when two records grade the same finding; conflicts are counted."""
        if current is None or new is None or current == new:
            return current if new is None else new if current is None else current
        order = self.allowed.get(name, {}).get("severity_order", [])
        if current in order and new in order:
            self._count(self.merged, ("severity_conflict", name))
            return order[max(order.index(current), order.index(new))]
        return current

    def resolve_uncertain(self, case_id, payload):
        """A positive and an uncertain record from the same quote: the uncertain one stands.

        The matching quote leaves the positive row; the row itself stays
        when other quotes support it. An uncertain record whose quote
        differs from every quote of the positive is dropped in favour of
        the positive.
        """
        for u in payload.get("uncertain", []):
            positive = (case_id, u["organ"], u["text_source"], "condition", u["condition"])
            row = self.rows.get(positive)
            if row is None:
                continue
            target = normalised(u.get("supporting_quote"))
            remaining = [q for q in row["_quotes"] if normalised(q) != target]
            if len(remaining) == len(row["_quotes"]):
                self._count(self.dropped,
                            ("uncertain_after_positive", u["condition"], u["organ"]))
                uncertain_key = (case_id, u["organ"], u["text_source"], "uncertain", u["condition"])
                self.rows.pop(uncertain_key, None)
            elif remaining:
                row["_quotes"] = remaining
                self._count(self.merged, ("positive_kept_on_other_quotes", u["condition"]))
            else:
                del self.rows[positive]
                self._count(self.dropped,
                            ("positive_masked_by_uncertain", u["condition"], u["organ"]))

    def mark_bilateral(self):
        """Flag a record whose twin for the paired organ has the same name and quotes.

        Quotes are compared as sets after normalisation; both twins are
        flagged, so a consumer picks the primary side itself.
        """
        for (case_id, organ, text_source, kind, name), row in self.rows.items():
            other = PAIRED_ORGANS.get(organ)
            if other is None:
                continue
            twin = self.rows.get((case_id, other, text_source, kind, name))
            if twin is None:
                continue
            twin_quotes = {normalised(q) for q in twin["_quotes"]}
            if twin_quotes == {normalised(q) for q in row["_quotes"]}:
                row["bilateral_replicated"] = True

    def table(self):
        modifier_columns = [f"mod_{key}" for key in sorted(self.modifier_keys)]
        if not self.rows:
            return pd.DataFrame(columns=BASE_COLUMNS + modifier_columns)
        for row in self.rows.values():
            mods = row.pop("_mods")
            for key in self.modifier_keys:
                row[f"mod_{key}"] = mods.get(key)
            row["finding_examples"] = " | ".join(row.pop("_quotes"))
        table = pd.DataFrame(self.rows.values())
        return table[BASE_COLUMNS + modifier_columns].reset_index(drop=True)


def aggregate_findings(per_case, schema=None):
    """The findings table from ``{case_id: payload}`` and the counts of corrections.

    A finding marked ``negated`` (a field of earlier schema versions) is
    skipped. With a schema, a record whose organ the finding does not allow
    is dropped, and a severity, extent or modifier value outside the
    finding's own lists is nulled; every such correction is counted. Within
    one (case, organ, text source, kind, finding) key the higher severity
    wins, the first extent and modifier values win, and distinct quotes
    are collected up to ``MAX_QUOTES``. An uncertain record whose quote
    also produced a positive masks that quote, and the positive when no
    other quote supports it; an uncertain record with a different quote is
    dropped in favour of the positive. Returns ``(table, counts)``.
    """
    agg = Aggregation(schema)
    for case_id, payload in per_case.items():
        for list_name, kind, name_key in RECORD_LISTS:
            for record in payload.get(list_name, []):
                if list_name == "findings" and record.get("negated"):
                    continue
                agg.ingest(case_id, record, kind, name_key)
        agg.resolve_uncertain(case_id, payload)
    agg.mark_bilateral()
    unknown = {k: v for k, v in agg.dropped.items() if k[0] == "unknown_text_source"}
    if unknown:
        warnings.warn(f"records with an unknown text_source dropped: {unknown}", stacklevel=2)
    counts = {
        "dropped": {" / ".join(map(str, k)): v for k, v in sorted(agg.dropped.items())},
        "nulled": {" / ".join(map(str, k)): v for k, v in sorted(agg.nulled.items())},
        "merged": {" / ".join(map(str, k)): v for k, v in sorted(agg.merged.items())},
    }
    return agg.table(), counts
