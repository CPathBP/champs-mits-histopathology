"""Causal chains of the cause of death: reading ICD-11 codes and matching codes to map fields."""

import pandas as pd

from champs_pipeline.data_prep import decode_results as dr


def test_matcher_covers_categories_subcodes_and_ranges():
    matches = dr.icd10_matcher("J18, B50.9, J13-J16, P24.1-P24.9")
    for code in ["J18", "J18.9", "B50.9", "B50.91", "J13", "J15.1", "J16", "P24.1", "P24.9"]:
        assert matches(code), code
    for code in ["J19", "B50", "B50.0", "J12.1", "J17", "P24.0", "P24"]:
        assert not matches(code), code


def test_icd11_codes_are_read_through_the_correspondence_or_the_description():
    descriptions = {
        "J18": "Pneumonia",
        "CA40": "Pneumonia",
        "KA60": "Sepsis of foetus or newborn",
        "P36": "Bacterial sepsis of newborn",
    }
    readings = dr.icd10_readings({"J18.9", "CA40.Z", "KA60", "XY99"}, descriptions,
                                 {"KA60": "P36"})
    assert readings["J18.9"] == ["J18.9"]
    assert readings["KA60"] == ["P36"]
    assert readings["CA40.Z"] == ["J18"]
    assert readings["XY99"] == []


def test_numeric_icd11_codes_are_restored(tmp_path):
    path = tmp_path / "icd_descriptions.csv"
    pd.DataFrame({"ICD_code": ["1.00E+30", "J18"], "ICD_desc": ["Influenza", "Pneumonia"],
                  "ICDRevision": [11, 10]}).to_csv(path, index=False)
    assert dr.load_icd_descriptions(path) == {"1E30": "Influenza", "J18": "Pneumonia"}


def test_every_position_of_the_chain_is_searched():
    chains = pd.DataFrame({column: [""] * 3 for column in dr.CHAIN_COLUMNS}, index=["a", "b", "c"])
    chains.loc["a", "Underlying_Cause"] = "J18"
    chains.loc["b", "Morbid_Condition_08"] = "J18.9"
    readings = dr.icd10_readings(dr.chain_codes(chains), {}, {})
    codes = dr.matching_codes("J18", readings)
    assert dr.cases_with_codes(chains, codes).tolist() == [True, True, False]
