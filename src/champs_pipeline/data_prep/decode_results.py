"""The panel-assigned cause of death (DeCoDe) per case, from the release table.

Besides the cause groups of each case, the module reads the causal chain of
each case (underlying cause, immediate cause and morbid conditions 1 to 8)
and matches its codes against ICD-10 codes and ranges. Codes of the chain
are ICD-10 or ICD-11. An ICD-11 code is read as ICD-10 through a curated
correspondence, or else through an identical description in the release's
ICD table; an ICD-11 code with neither is not matched.
"""

import re

import pandas as pd

COLUMNS = ["champs_deid", "UC_champs_group_desc", "IC_champs_group_desc"]
MORBID_CONDITIONS = [f"Morbid_Condition_0{number}" for number in range(1, 9)]
CHAIN_COLUMNS = ["Underlying_Cause", "Immediate_COD", *MORBID_CONDITIONS]

# A letter, two digits and an optional subcode: J18, B50.9, P24.1.
ICD10_CODE = re.compile(r"^([A-Z])(\d\d)(?:\.(\d+))?$")
# ICD-11 codes such as 1E30 that a spreadsheet stored as a number (1.00E+30).
NUMERIC_ICD11_CODE = r"^(\d)\.00E\+(\d+)$"


def load_decode_results(path):
    """One row per case: the underlying and the immediate cause group."""
    table = pd.read_csv(path, usecols=COLUMNS, low_memory=False).drop_duplicates("champs_deid")
    table["champs_deid"] = table["champs_deid"].astype(str)
    return table.rename(columns={"UC_champs_group_desc": "underlying_cause_group",
                                 "IC_champs_group_desc": "immediate_cause_group"})


def normalise_code(value):
    """The code in upper case without spaces; an empty string for a missing value."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value)).upper()


def load_cause_chains(path, case_ids):
    """The causal chain of each listed case that has a DeCoDe record, indexed by case."""
    table = pd.read_csv(path, usecols=["champs_deid", *CHAIN_COLUMNS], low_memory=False)
    table = table.drop_duplicates("champs_deid")
    table["champs_deid"] = table["champs_deid"].astype(str)
    table = table[table["champs_deid"].isin(set(case_ids))].set_index("champs_deid")
    for column in CHAIN_COLUMNS:
        table[column] = table[column].map(normalise_code)
    return table


def load_icd_descriptions(path):
    """Code to description, from the release's ICD table, with numeric ICD-11 codes restored."""
    table = pd.read_csv(path, low_memory=False)
    codes = table["ICD_code"].astype(str).str.strip().str.upper()
    codes = codes.str.replace(NUMERIC_ICD11_CODE, r"\1E\2", regex=True)
    return dict(zip(codes, table["ICD_desc"].fillna("").astype(str)))


def load_icd11_correspondence(path):
    """ICD-11 code to ICD-10 code, from the curated correspondence table."""
    table = pd.read_csv(path, dtype=str)
    return dict(zip(table["icd11_code"], table["icd10_code"]))


def chain_codes(chains):
    """Every code that occurs in the causal chains."""
    codes = set()
    for column in CHAIN_COLUMNS:
        codes |= set(chains[column])
    codes.discard("")
    return codes


def icd10_readings(codes, descriptions, icd11_correspondence):
    """For each code, the ICD-10 codes it is read as; an empty list if it cannot be read.

    An ICD-10 code is read as itself. An ICD-11 code is read through the curated
    correspondence or, failing that, as every ICD-10 code whose description in the
    release's ICD table is identical to its own (or, for a subcode without a
    description, to that of its category).
    """
    icd10_by_description = {}
    for code, description in descriptions.items():
        if ICD10_CODE.match(code) and description:
            icd10_by_description.setdefault(description.strip().lower(), []).append(code)
    readings = {}
    for code in sorted(codes):
        if ICD10_CODE.match(code):
            readings[code] = [code]
        elif code in icd11_correspondence:
            readings[code] = [icd11_correspondence[code]]
        else:
            description = descriptions.get(code) or descriptions.get(code.split(".")[0]) or ""
            readings[code] = sorted(icd10_by_description.get(description.strip().lower(), []))
    return readings


def range_test(part):
    """The test for an ICD-10 range of categories (J13-J16) or of subcodes (P24.1-P24.9)."""
    low = ICD10_CODE.match(part.split("-")[0].strip())
    high = ICD10_CODE.match(part.split("-")[1].strip())
    if low is None or high is None or low.group(1) != high.group(1):
        raise ValueError(f"unsupported ICD-10 range: {part}")
    if low.group(3) is None and high.group(3) is None:
        def in_category_range(code):
            return (code.group(1) == low.group(1)
                    and int(low.group(2)) <= int(code.group(2)) <= int(high.group(2)))
        return in_category_range

    def in_subcode_range(code):
        if code.group(1) != low.group(1) or code.group(2) != low.group(2) or code.group(3) is None:
            return False
        return int(low.group(3)) <= int(code.group(3)[0]) <= int(high.group(3))
    return in_subcode_range


def code_test(part):
    """The test for one ICD-10 code: the code itself and all of its subcodes."""
    single = ICD10_CODE.match(part.strip())
    if single is None:
        raise ValueError(f"unsupported ICD-10 code: {part}")

    def under_code(code):
        if code.group(1) != single.group(1) or code.group(2) != single.group(2):
            return False
        if single.group(3) is None:
            return True
        return (code.group(3) or "").startswith(single.group(3))
    return under_code


def icd10_matcher(field):
    """A function that tells whether an ICD-10 code falls under a field of codes and ranges.

    The field separates its parts by commas. ``J18`` covers the category and all of
    its subcodes, ``B50.9`` the code and its subcodes, ``J13-J16`` the categories J13
    to J16, and ``P24.1-P24.9`` the subcodes P24.1 to P24.9.
    """
    tests = []
    for part in re.split(r"[,+]", field):
        if not part.strip():
            continue
        if "-" in part:
            tests.append(range_test(part))
        else:
            tests.append(code_test(part))

    def matches(icd10_code):
        parsed = ICD10_CODE.match(icd10_code)
        return parsed is not None and any(test(parsed) for test in tests)
    return matches


def matching_codes(field, readings):
    """The chain codes with at least one ICD-10 reading under the field."""
    if pd.isna(field) or not str(field).strip():
        return set()
    matches = icd10_matcher(str(field))
    hits = set()
    for code, icd10_codes in readings.items():
        if any(matches(icd10_code) for icd10_code in icd10_codes):
            hits.add(code)
    return hits


def cases_with_codes(chains, codes):
    """For each case, whether any position of its causal chain holds one of the codes."""
    found = pd.Series(False, index=chains.index)
    for column in CHAIN_COLUMNS:
        found |= chains[column].isin(codes)
    return found
