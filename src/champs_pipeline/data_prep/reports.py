"""Parsing of the pathology report export.

The report of a case is the ``HEText`` field of the diagnosis table, often
with HTML markup. This module cleans the text, finds the three report
sections, classifies the reporting template and the origin of the text
(explicit sections, headerless, or empty), and tells, for a slide file
name, whether the slide was scanned at the site or at the central
pathology laboratory (CPL). All functions are pure.

Text sources and slide sources: a finding is written under one of three
section headers or in a headerless report. ``site_report`` and
``cpl_review_of_scans`` describe the slides scanned at the site;
``cpl_slides`` describes the slides cut and scanned at the CPL. Headerless
reports read as CPL-slide descriptions in an audit of their vocabulary and
are routed as ``implicit_cpl``.
"""

import html
import re

import pandas as pd

TEXT_SOURCE_TO_SLIDE_SOURCE = {
    "site_report": "SITE",
    "cpl_review_of_scans": "SITE",
    "cpl_slides": "CPL",
    "implicit_cpl": "CPL",
}
TEXT_SOURCES = tuple(TEXT_SOURCE_TO_SLIDE_SOURCE)

# Section headers. The SITE header has a `PATH` spelling variant; the CPL
# review header is consumed whole so its trailing `SCANNED SLIDES` does not
# leak into the section body; the CPL slides header must start a line so
# that the word `SLIDES` inside the review header cannot trigger it.
_RE_SITE = re.compile(
    r"SITE\s+(?:PATH\s+)?REPORT\s+(?:PATHOLOGY\s+)?MAJOR\s+FINDINGS|SITE\s+PATHOLOGY",
    re.IGNORECASE,
)
_RE_REVIEW = re.compile(r"CPL\s+REVIEW(?:\s+OF\s+SITE(?:\s+SCANN?ED\s+\w+)?)?", re.IGNORECASE)
_RE_CPL_SLIDES = re.compile(
    r"^\s*CPL\s+SLIDES?(?:\s+REVIEW)?(?:\s*:|\s*$)", re.IGNORECASE | re.MULTILINE
)
# Any header line. The CPL slides alternative needs a colon or the end of
# the line, so a prose line that starts with "CPL slides ..." is not a header.
_RE_ANY_HDR = re.compile(
    r"^[ \t]*(?:SITE\s+(?:PATH\s+)?REPORT\b|SITE\s+PATHOLOGY\b|CPL\s+REVIEW\b|"
    r"CPL\s+SLIDES?(?:\s+REVIEW)?\s*(?::|$))",
    re.IGNORECASE | re.MULTILINE,
)
_RE_HTML_TAG = re.compile(r"<div|<font", re.IGNORECASE)
_RE_BLOCK_TAG = re.compile(
    r"</?(?:div|font|p|br|tr|td|th|li|ul|ol|table|thead|tbody|h[1-6]|hr|blockquote)\b[^>]*>",
    re.IGNORECASE,
)
_RE_INLINE_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

# A slide scanned at the CPL carries the accession prefix `YYYY-NNNN-X_`.
# The separators vary between `_` and `-`, the section code can be
# alphanumeric (`FS3`, `IHC2022`), and spaces occur around the section letter.
_RE_CPL_PREFIX = re.compile(r"^\d{4}[-_]\d{3,4}[-_]\s*[A-Za-z][A-Za-z0-9]*\s*[-_]")

# Phrases characteristic of a CPL description of physical slides.
CPL_SIGNATURE_PATTERNS = (
    r"\btechnically\s+adequate\b",
    r"\bcore\s+fragments?\s+contain\b",
    r"\b\d+\s+cores?\s+contain\b",
    r"\b\d+\s+of\s+\d+\s+(?:tissue\s+)?(?:cores?|fragments?|core\s+fragments?)\b",
    r"\bother\s+tissues?\s+present\b",
    r"\bdiagnostically\s+(?:ir)?relevant\b",
    r"\bCPL\s+slides?\s+not\s+examined\b",
)
_RE_ORGAN_LINE = re.compile(r"(liver|lung|cns|brain|placenta|kidney)\s*:", re.IGNORECASE)
_RE_EMPTY_SECTION = re.compile(
    r"\s*(not provided|not performed|not reviewed|not available|not examined|"
    r"not done|no tissue collected|no data available|no slides?(?: available)?|"
    r"nil|deferred|pending|none received|n/a|na|none|-+)\s*\.?\s*"
)
_RE_ORGAN_FINDING = re.compile(
    r"(livers?|lungs?|cns|brains?|placentas?|kidneys?|bone marrow|hearts?)\s*[:,]"
)

ORIGIN_LABELS = ("EXPLICIT", "HEADERLESS", "EMPTY")


def strip_hetext_html(text):
    """Remove HTML markup from a report field and normalise its whitespace.

    Block-level tags become line breaks so the structure survives; inline
    tags are removed without a break so a tag inside a word does not split
    it. A literal ``<`` before a digit or a space is not a tag.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = _RE_BLOCK_TAG.sub("\n", str(text))
    s = _RE_INLINE_TAG.sub("", s)
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"\n+", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def has_html_markup(raw_text):
    """Whether the raw report field contains ``<div>`` or ``<font>`` markup."""
    if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text)):
        return False
    return bool(_RE_HTML_TAG.search(str(raw_text)))


def extract_sections(text):
    """Split a cleaned report into its three sections.

    Returns a dict with the keys ``SITE_REPORT``, ``CPL_REVIEW`` and
    ``CPL_SLIDES``; a missing section maps to ``None``. A section ends at
    the next header of any kind or at the end of the text.
    """
    sections = {"SITE_REPORT": None, "CPL_REVIEW": None, "CPL_SLIDES": None}
    if not text:
        return sections
    headers = (
        ("SITE_REPORT", _RE_SITE), ("CPL_REVIEW", _RE_REVIEW), ("CPL_SLIDES", _RE_CPL_SLIDES),
    )
    for key, regex in headers:
        match = regex.search(text)
        if not match:
            continue
        start = match.end()
        next_starts = []
        for _, other in headers:
            following = other.search(text, pos=start)
            if following:
                next_starts.append(following.start())
        end = min(next_starts) if next_starts else len(text)
        sections[key] = text[start:end].strip()
    return sections


def is_meaningful_section(section_text):
    """Whether a section carries organ findings rather than a placeholder."""
    if not section_text:
        return False
    lowered = section_text.lower()
    if _RE_EMPTY_SECTION.fullmatch(lowered):
        return False
    return bool(_RE_ORGAN_FINDING.search(lowered))


def cpl_signature_score(text):
    """Number of distinct CPL-slide phrases that occur in the text."""
    if not text:
        return 0
    hits = (re.search(pattern, text, re.IGNORECASE) for pattern in CPL_SIGNATURE_PATTERNS)
    return sum(1 for hit in hits if hit)


def classify_template(raw_hetext, cleaned_hetext=None, *, has_dxagent=False,
                      has_dxsystem=False, has_dxsyndrome=False):
    """The reporting template of one report row.

    ``HTML`` when the field carries markup; ``DeCoDe`` when the structured
    diagnosis fields are filled and the text has no section headers;
    ``plain_NoHdr`` for plain text without headers; ``plain_Hdr`` for plain
    text with headers; ``empty`` when there is no text.
    """
    if cleaned_hetext is None:
        cleaned_hetext = strip_hetext_html(raw_hetext)
    if not cleaned_hetext:
        return "empty"
    has_any_header = bool(_RE_ANY_HDR.search(cleaned_hetext))
    has_dx = has_dxagent or has_dxsystem or has_dxsyndrome
    if has_html_markup(raw_hetext):
        return "HTML"
    if has_dx and not has_any_header:
        return "DeCoDe"
    if not has_any_header:
        return "plain_NoHdr"
    return "plain_Hdr"


def classify_slide_source(filename):
    """``CPL`` for a slide file with a CPL accession prefix, ``SITE`` otherwise."""
    if not filename:
        return "SITE"
    base = filename.rsplit("/", 1)[-1]
    return "CPL" if _RE_CPL_PREFIX.match(base) else "SITE"


def attribute_origin(cleaned_hetext):
    """The origin of a cleaned report row: ``EXPLICIT``, ``HEADERLESS`` or ``EMPTY``.

    Explicit: section headers with at least one meaningful section.
    Headerless: no headers, but an organ line or a CPL-slide phrase; such
    reports read as CPL slide descriptions. Empty: neither, or headers
    whose sections are all placeholders.
    """
    if not cleaned_hetext:
        return "EMPTY"
    if _RE_ANY_HDR.search(cleaned_hetext):
        sections = extract_sections(cleaned_hetext)
        if any(is_meaningful_section(text) for text in sections.values()):
            return "EXPLICIT"
        return "EMPTY"
    if _RE_ORGAN_LINE.search(cleaned_hetext) or cpl_signature_score(cleaned_hetext) >= 1:
        return "HEADERLESS"
    return "EMPTY"
