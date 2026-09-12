"""Study and case identifiers parsed from slide names and paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Pattern for CHAMPS study IDs: 2-5 uppercase letters followed by 4-6 digits.
# Requires underscore or start-of-field before the ID to avoid matching IHC batch
# numbers like IHC2017-069-21 (where "IHC20170" would be a false positive).
# Negative lookbehind excludes IHC prefix (IHC batch numbers use IHC+year format).
# Examples: BDAA00001, KEAA12345, MZCC00048, ZAAAA1765, MZAA0046, ETA000052
STUDY_ID_PATTERN = re.compile(r"(?:^|[_/])(?!IHC\d)([A-Z]{2,5}\d{4,6})(?=[_/.,\s-]|$)")


def study_id_in_name(slide_id: str) -> Optional[str]:
    """The study id embedded in a slide name, or None when the name carries none."""
    match = STUDY_ID_PATTERN.search(str(slide_id))
    return match.group(1) if match else None


def derive_case_id(wsi_path: str) -> str:
    """Derive case ID from a WSI file path.

    Extracts the folder name after "Cases/" in the path, or falls back
    to the parent directory name.

    Parameters
    ----------
    wsi_path : str
        Path to a WSI file.

    Returns
    -------
    str
        The derived case ID.
    """
    p = Path(wsi_path)
    parts = p.parts
    if "Cases" in parts:
        i = parts.index("Cases")
        if i + 1 < len(parts):
            return str(parts[i + 1])
    return p.parent.name
