"""The extraction schema: the findings a report can carry and their qualifiers.

The schema file (``configs/extraction/morphology_schema_<version>.yaml``)
lists the organs, the text sources, the positive findings (``conditions``),
the negation flags, the quality flags and the clinical diagnoses the
language model may emit, with the severity and extent ladders and the
modifiers of each finding. This module loads it, fingerprints it, and
projects it into the two forms the extraction uses: the dict the prompt
template renders, and the JSON schema that constrains the model's reply.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ALLOWED_ORGANS = {"right_lung", "left_lung", "liver", "cns_post", "cns_trans"}

# Reasons an ``uncertain`` record may carry; the prompt defines their use.
UNCERTAIN_REASONS = ("hedge", "material_unstated", "pathologist_uncertainty")


@dataclass(frozen=True)
class ConditionSpec:
    """One positive finding, with its ladders and modifiers."""

    name: str
    organs: tuple
    description: str
    synonyms: tuple
    severity_values: tuple
    extent_values: tuple
    modifiers: dict


@dataclass(frozen=True)
class NegationFlagSpec:
    """One statement of absence the model may record."""

    name: str
    organs: tuple
    description: str
    synonyms: tuple


@dataclass(frozen=True)
class QualityFlagSpec:
    """One sample-quality marker (autolysis, inadequate tissue, ...)."""

    name: str
    organs: tuple
    description: str
    synonyms: tuple
    severity_values: tuple
    extent_values: tuple
    modifiers: dict


@dataclass(frozen=True)
class DiagnosisSpec:
    """One clinical diagnosis line the model records without a morphology."""

    name: str
    organs: tuple
    description: str
    synonyms: tuple


@dataclass(frozen=True)
class MorphologySchema:
    """The loaded schema; ``schema_sha`` fingerprints the file's bytes."""

    schema_version: str
    schema_sha: str
    organs: tuple
    text_sources: tuple
    conditions: tuple
    negation_flags: tuple
    quality_flags: tuple
    diagnoses: tuple = ()
    _by_name: dict = field(repr=False, default_factory=dict)

    def spec(self, name):
        """The entry of any section with this name, or ``None``."""
        return self._by_name.get(name)


def _strings(values):
    """A tuple of strings from a YAML list, empty when the list is missing."""
    if not values:
        return ()
    return tuple(str(v) for v in values)


def _organs(name, organs):
    bad = [o for o in organs if o not in ALLOWED_ORGANS]
    if bad:
        raise ValueError(f"schema entry {name!r}: unknown organs {bad}")
    return tuple(organs)


def _modifiers(raw):
    """``{modifier: (allowed values...)}`` from the YAML block, empty when absent."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"modifiers must be a mapping, got {type(raw).__name__}")
    return {str(key): _strings(values) for key, values in raw.items()}


def load_schema(path):
    """Load and validate a schema file; unknown organs or duplicate names raise."""
    text = Path(path).read_text()
    raw = yaml.safe_load(text)
    schema_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    conditions = tuple(
        ConditionSpec(
            name=entry["name"],
            organs=_organs(entry["name"], entry.get("organs", [])),
            description=entry.get("description", ""),
            synonyms=_strings(entry.get("synonyms")),
            severity_values=_strings(entry.get("severity_values")),
            extent_values=_strings(entry.get("extent_values")),
            modifiers=_modifiers(entry.get("modifiers")),
        )
        for entry in raw.get("conditions", [])
    )
    negation_flags = tuple(
        NegationFlagSpec(
            name=entry["name"],
            organs=_organs(entry["name"], entry.get("organs", [])),
            description=entry.get("description", ""),
            synonyms=_strings(entry.get("synonyms")),
        )
        for entry in raw.get("negation_flags", [])
    )
    diagnoses = tuple(
        DiagnosisSpec(
            name=entry["name"],
            organs=_organs(entry["name"], entry.get("organs", [])),
            description=entry.get("description", ""),
            synonyms=_strings(entry.get("synonyms")),
        )
        for entry in raw.get("diagnoses", [])
    )
    quality_flags = tuple(
        QualityFlagSpec(
            name=entry["name"],
            organs=_organs(entry["name"], entry.get("organs", [])),
            description=entry.get("description", ""),
            synonyms=_strings(entry.get("synonyms")),
            severity_values=_strings(entry.get("severity_values")),
            extent_values=_strings(entry.get("extent_values")),
            modifiers=_modifiers(entry.get("modifiers")),
        )
        for entry in raw.get("quality_flags", [])
    )

    entries = conditions + negation_flags + quality_flags + diagnoses
    names = [spec.name for spec in entries]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"schema names declared more than once: {duplicates}")

    return MorphologySchema(
        schema_version=str(raw.get("schema_version", "")),
        schema_sha=schema_sha,
        organs=_strings(raw.get("organs")),
        text_sources=_strings(raw.get("text_sources")),
        conditions=conditions,
        negation_flags=negation_flags,
        quality_flags=quality_flags,
        diagnoses=diagnoses,
        _by_name={spec.name: spec for spec in entries},
    )


def build_schema_view(schema):
    """The dict the prompt template renders as tables and lists."""
    return {
        "organs": list(schema.organs),
        "text_sources": list(schema.text_sources),
        "conditions": [
            {
                "name": c.name, "organs": c.organs, "description": c.description,
                "synonyms": c.synonyms, "severity_values": c.severity_values,
                "extent_values": c.extent_values, "modifiers": c.modifiers,
            }
            for c in schema.conditions
        ],
        "negation_flags": [
            {"name": n.name, "organs": n.organs, "description": n.description,
             "synonyms": n.synonyms}
            for n in schema.negation_flags
        ],
        "quality_flags": [
            {
                "name": q.name, "organs": q.organs, "description": q.description,
                "severity_values": q.severity_values, "extent_values": q.extent_values,
                "modifiers": q.modifiers,
            }
            for q in schema.quality_flags
        ],
        "diagnoses": [
            {"name": d.name, "organs": d.organs, "description": d.description,
             "synonyms": d.synonyms}
            for d in schema.diagnoses
        ],
    }


def _enum(values):
    return {"type": "string", "enum": sorted(values)}


def _enum_or_null(values):
    return {"anyOf": [_enum(values), {"type": "null"}]}


def _record(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def output_json_schema(schema):
    """The JSON schema of one case's reply, used to constrain decoding.

    Enumerations are the unions over the schema: every severity value, every
    extent value and, per modifier key, every value any finding allows. The
    per-finding ladders are narrower and are enforced afterwards, when the
    records are aggregated (a value outside a finding's ladder is nulled).
    """
    graded = schema.conditions + schema.quality_flags
    severities = {v for spec in graded for v in spec.severity_values}
    extents = {v for spec in graded for v in spec.extent_values}
    modifier_values = {}
    for spec in graded:
        for key, values in spec.modifiers.items():
            modifier_values.setdefault(key, set()).update(values)
    modifiers = {
        "type": "object",
        "properties": {key: _enum_or_null(modifier_values[key]) for key in sorted(modifier_values)},
        "additionalProperties": False,
    }
    organ = _enum(schema.organs)
    text_source = _enum(schema.text_sources)
    condition = _enum(c.name for c in schema.conditions)
    quote = {"type": "string"}

    finding = _record({
        "organ": organ,
        "text_source": text_source,
        "condition": condition,
        "severity": _enum_or_null(severities),
        "extent": _enum_or_null(extents),
        "modifiers": modifiers,
        "supporting_quote": quote,
    })
    negation = _record({
        "organ": organ,
        "text_source": text_source,
        "negation_group": _enum(n.name for n in schema.negation_flags),
        "supporting_quote": quote,
    })
    quality_flag = _record({
        "organ": organ,
        "text_source": text_source,
        "flag": _enum(q.name for q in schema.quality_flags),
        "severity": _enum_or_null(severities),
        "extent": _enum_or_null(extents),
        "modifiers": modifiers,
        "supporting_quote": quote,
    })
    uncertain = _record({
        "organ": organ,
        "text_source": text_source,
        "condition": condition,
        "reason": _enum(UNCERTAIN_REASONS),
        "supporting_quote": quote,
    })
    reply = {
        "case_id": {"type": "string"},
        "findings": {"type": "array", "items": finding},
        "negations": {"type": "array", "items": negation},
        "quality_flags": {"type": "array", "items": quality_flag},
        "uncertain": {"type": "array", "items": uncertain},
    }
    if schema.diagnoses:
        diagnosis = _record({
            "organ": organ,
            "text_source": text_source,
            "diagnosis": _enum(d.name for d in schema.diagnoses),
            "supporting_quote": quote,
        })
        reply["diagnoses"] = {"type": "array", "items": diagnosis}
    return _record(reply)
