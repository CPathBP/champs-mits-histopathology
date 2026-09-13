"""MITS tissue codes parsed from slide names.

A slide name carries the specimen number and the tissue code, three
digits in most names (``M00038.045``, ``M01168_043``) and two in some
early ones (``M00015.41``). Odd codes are the primary specimen and even
codes the backup; both name the same organ.
"""

import re

# The code follows the specimen number (an M and digits).
_TISSUE_CODE = re.compile(r"M\d+[._](0\d{2}|\d{2})(?=[._\s\-(]|$)")

ORGAN_OF_CODE = {
    "041": "liver", "042": "liver",
    "043": "right_lung", "044": "right_lung",
    "045": "left_lung", "046": "left_lung",
    "047": "cns_post", "048": "cns_post",
    "049": "cns_trans", "050": "cns_trans",
}
ORGAN_NAME = {
    "right_lung": "Right lung", "left_lung": "Left lung", "liver": "Liver",
    "cns_post": "CNS posterior", "cns_trans": "CNS transnasal",
}


def tissue_code(slide_id):
    """The tissue code in a slide name as three digits, or None."""
    match = _TISSUE_CODE.search(str(slide_id))
    return match.group(1).zfill(3) if match else None


def organ_of_slide(slide_id):
    """The organ the slide's tissue code names, or None."""
    code = tissue_code(slide_id)
    return ORGAN_OF_CODE.get(code) if code else None
