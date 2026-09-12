"""Stain detection from slide names.

The archive names a slide after its stain when it is not H&E, either as a
whole word (``_GRAM_``, ``_PAS_``) or, for immunohistochemistry, as ``IHC``
followed by a batch number (``_IHC2019-227-25_``). A name without a stain
token is H&E.
"""

import re
from typing import Optional

STAIN_NORMALIZATION = {
    "H_AND_E": "HE", "HE": "HE", "H_E": "HE",
    "GRAMSTAIN": "GRAM", "GRAM": "GRAM", "GROCOTT": "GROCOTT",
    "TRICHROME": "TRICHROME", "GMS": "GMS", "PAS": "PAS", "AFB": "AFB", "IRON": "IRON", "ZN": "ZN",
    "RETICULIN": "RETICULIN", "WS": "WS", "VERHOEFF": "VERHOEFF", "MUCICARMINE": "MUCICARMINE",
    "HALL": "HALL", "FONTANA": "FONTANA", "FM": "FM", "IHC": "IHC",
}

# Longest tokens first, so that GRAMSTAIN matches before GRAM. IHC may be
# followed by its batch number; every other token must stand alone.
_WORD_TOKENS = sorted((t for t in STAIN_NORMALIZATION if t != "IHC"), key=len, reverse=True)
_STAIN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(IHC(?![A-Za-z])|(?:"
    + "|".join(map(re.escape, _WORD_TOKENS))
    + r")(?![A-Za-z0-9]))",
    re.IGNORECASE,
)


def detect_stain(slide_id: str) -> Optional[str]:
    """The first stain token in a slide id (``IHC``, ``GRAM``, ...), or None."""
    match = _STAIN_PATTERN.search(str(slide_id))
    return STAIN_NORMALIZATION[match.group(1).upper()] if match else None


def is_he_slide(slide_id: str, predicted_stain: Optional[str] = None) -> bool:
    """H&E by name (no stain token, or an H&E token) and, when given, by the predicted stain."""
    if predicted_stain is not None and str(predicted_stain).strip() not in ("", "HE", "nan"):
        return False
    return detect_stain(slide_id) in (None, "HE")
