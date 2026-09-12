"""Tests of the report arm: parsing, corpus rule, aggregation, and the JSON schema."""

import json

import pandas as pd
import pytest

from champs_pipeline.data_prep.funnel import build_case_attribution
from champs_pipeline.data_prep.llm_aggregate import aggregate_findings, copy_concurred_site_records
from champs_pipeline.data_prep.morphology_labels import load_schema, output_json_schema
from champs_pipeline.data_prep.reports import (
    attribute_origin,
    classify_slide_source,
    classify_template,
    extract_sections,
    has_html_markup,
    is_meaningful_section,
    strip_hetext_html,
)

import build_corpus
import extract_findings
import finalize_extraction
import score_against_annotations

THREE_SECTIONS = (
    "SITE REPORT MAJOR FINDINGS:\nLiver: unremarkable\n"
    "CPL REVIEW OF SITE SCANNED SLIDES:\nLiver: mild EMH\n"
    "CPL SLIDES:\nLiver: 5 cores; technically adequate."
)


def test_strip_hetext_html():
    raw = "<div><font color=black>Liver: unremarkable&nbsp;</font></div>"
    assert strip_hetext_html(raw) == "Liver: unremarkable"
    assert strip_hetext_html("<strong>L</strong>iver: <2 cores") == "Liver: <2 cores"
    assert strip_hetext_html(None) == ""
    assert has_html_markup(raw) and not has_html_markup("plain")


def test_extract_sections():
    sections = extract_sections(THREE_SECTIONS)
    assert "Liver: unremarkable" in sections["SITE_REPORT"]
    assert "mild EMH" in sections["CPL_REVIEW"]
    assert "5 cores" in sections["CPL_SLIDES"]
    assert extract_sections("Liver: unremarkable") == {
        "SITE_REPORT": None, "CPL_REVIEW": None, "CPL_SLIDES": None,
    }


def test_is_meaningful_section():
    assert is_meaningful_section("Liver: unremarkable")
    assert not is_meaningful_section("Not performed")
    assert not is_meaningful_section("CPL slides not examined for other tissues")
    assert not is_meaningful_section(None)


def test_classify_template():
    assert classify_template("<div>CPL REVIEW OF SITE SCANNED SLIDES:</div>") == "HTML"
    assert classify_template("SITE REPORT MAJOR FINDINGS:\nLiver: ok") == "plain_Hdr"
    assert classify_template("Liver: 6 cores. Unremarkable.", has_dxagent=True) == "DeCoDe"
    assert classify_template("Liver: 6 cores. Unremarkable.") == "plain_NoHdr"
    assert classify_template(None) == "empty"


@pytest.mark.parametrize("filename,expected", [
    ("M00982.043_BDAA00751.ndpi", "SITE"),
    ("M10186.043 - 2025-08-15 13.11.25.ndpi", "SITE"),
    ("2023-1176-A_HE_KEAA02093_M07233.042.svs", "CPL"),
    ("2022-1235-B_IHC2022-319-20_ZAAA02301_M05405.044.svs", "CPL"),
    ("Bangladesh/Cases/BDAA/2023-0001-A_HE_BDAA00001_M0001.042.svs", "CPL"),
])
def test_classify_slide_source(filename, expected):
    assert classify_slide_source(filename) == expected


def test_attribute_origin():
    assert attribute_origin(THREE_SECTIONS) == "EXPLICIT"
    headerless = "Liver: 6 of 6 core fragments contain liver; technically adequate. Unremarkable."
    assert attribute_origin(headerless) == "HEADERLESS"
    placeholders = "SITE REPORT MAJOR FINDINGS:\nNot provided\nCPL SLIDES:\nNot performed"
    assert attribute_origin(placeholders) == "EMPTY"
    assert attribute_origin("") == "EMPTY"


def reports_table():
    return pd.DataFrame({
        "champs_deid": ["A", "A", "B", "C", None],
        "HEText": [THREE_SECTIONS, "<div>Liver: congestion</div>", None,
                   "Liver: 5 cores contain liver; technically adequate. Steatosis.", "x"],
        "DxAgentCode": [None, None, None, "E. coli", None],
        "DXSystemCode": [None] * 5,
        "DXSyndromeCode": [None] * 5,
    })


def test_corpus_cases_and_shards(tmp_path):
    reports_table().to_csv(tmp_path / "reports.csv", index=False)
    assert build_corpus.corpus_cases(tmp_path / "reports.csv") == ["A", "C"]
    shards = build_corpus.deal(["A", "B", "C", "D", "E"], 2, seed=0)
    assert sorted(shards[0] + shards[1]) == ["A", "B", "C", "D", "E"]
    assert len(shards[0]) == 3 and len(shards[1]) == 2


def test_case_attribution(tmp_path):
    reports_table().to_csv(tmp_path / "reports.csv", index=False)
    rows, cases = build_case_attribution(tmp_path / "reports.csv")
    assert len(rows) == 4 and list(cases["champs_deid"]) == ["A", "B", "C"]
    by_case = cases.set_index("champs_deid")
    assert by_case.loc["A", "template"] == "HTML"  # the highest-priority row wins
    assert by_case.loc["A", "has_hetext"]
    assert by_case.loc["B", "template"] == "empty" and not by_case.loc["B", "has_hetext"]
    assert by_case.loc["C", "template"] == "DeCoDe" and by_case.loc["C", "origin"] == "HEADERLESS"


def payloads():
    return {
        "A": {
            "findings": [
                {"organ": "right_lung", "text_source": "cpl_review_of_scans",
                 "condition": "bronchopneumonia", "severity": "moderate", "extent": "multifocal",
                 "modifiers": {"inflammation_character": "neutrophilic"},
                 "supporting_quote": "moderate multifocal bronchopneumonia"},
                {"organ": "right_lung", "text_source": "cpl_review_of_scans",
                 "condition": "bronchopneumonia", "severity": None, "extent": None,
                 "modifiers": {}, "supporting_quote": "patchy bronchopneumonia"},
                {"organ": "liver", "text_source": "cpl_slides", "condition": "steatosis",
                 "severity": "extensive", "extent": "focal", "modifiers": {},
                 "supporting_quote": "no steatosis", "negated": True},
                {"organ": "liver", "text_source": "cpl_slides", "condition": "steatosis",
                 "severity": "plenty", "extent": "focal",
                 "modifiers": {"droplet": "small_droplet", "acuity": "acute"},
                 "supporting_quote": "small droplet steatosis"},
                {"organ": "cns_post", "text_source": "cpl_slides", "condition": "steatosis",
                 "severity": None, "extent": None, "modifiers": {},
                 "supporting_quote": "steatosis in the brain"},
                {"organ": "right_lung", "text_source": "implicit_cpl",
                 "condition": "aspiration_squames", "severity": "severe", "extent": None,
                 "modifiers": {}, "supporting_quote": "Severe aspiration"},
                {"organ": "left_lung", "text_source": "cpl_slides",
                 "condition": "hyaline_membranes", "severity": None, "extent": None,
                 "modifiers": {}, "supporting_quote": "Lungs: hyaline membranes"},
                {"organ": "right_lung", "text_source": "cpl_slides",
                 "condition": "hyaline_membranes", "severity": None, "extent": None,
                 "modifiers": {}, "supporting_quote": "Lungs: hyaline membranes"},
            ],
            "negations": [{"organ": "liver", "text_source": "site_report",
                           "negation_group": "unremarkable", "supporting_quote": "Liver: ok"}],
            "quality_flags": [{"organ": "left_lung", "text_source": "unknown_source",
                               "flag": "autolysis", "supporting_quote": "autolysis"}],
            "uncertain": [
                {"organ": "right_lung", "text_source": "implicit_cpl",
                 "condition": "aspiration_squames", "reason": "material_unstated",
                 "supporting_quote": "Severe aspiration"},
                {"organ": "liver", "text_source": "cpl_slides", "condition": "cholestasis",
                 "reason": "hedge", "supporting_quote": "possible cholestasis"},
                {"organ": "liver", "text_source": "cpl_slides", "condition": "steatosis",
                 "reason": "hedge", "supporting_quote": "possible steatosis elsewhere"},
            ],
            "diagnoses": [{"organ": "right_lung", "text_source": "site_report",
                           "diagnosis": "malaria", "supporting_quote": "Malaria"}],
        },
    }


def one_row(table, name, kind):
    rows = table[(table["semantic_group"] == name) & (table["kind"] == kind)]
    assert len(rows) == 1, (name, kind, len(rows))
    return rows.iloc[0]


def test_aggregate_findings(schema_v45):
    with pytest.warns(UserWarning, match="unknown text_source"):
        table, counts = aggregate_findings(payloads(), schema=schema_v45)
    keys = set(zip(table["semantic_group"], table["kind"]))
    assert table["kind"].value_counts().to_dict() == {
        "condition": 4, "negation": 1, "uncertain": 2, "diagnosis": 1,
    }
    bpn = one_row(table, "bronchopneumonia", "condition")
    assert bpn["severity"] == "moderate" and bpn["slide_source"] == "SITE"
    quotes = "moderate multifocal bronchopneumonia | patchy bronchopneumonia"
    assert bpn["finding_examples"] == quotes
    assert bpn["mod_inflammation_character"] == "neutrophilic"
    steatosis = one_row(table, "steatosis", "condition")
    assert steatosis["organ"] == "liver"  # the cns_post record is off-organ and dropped
    assert steatosis["severity"] is None  # outside the ladder
    assert steatosis["extent"] == "focal" and steatosis["mod_droplet"] == "small_droplet"
    assert "mod_acuity" not in table.columns  # not a modifier of steatosis
    # An uncertain record with the same quote masks the positive; a different quote loses.
    assert ("aspiration_squames", "condition") not in keys
    assert one_row(table, "aspiration_squames", "uncertain")["mod_reason"] == "material_unstated"
    assert ("steatosis", "uncertain") not in keys
    assert one_row(table, "cholestasis", "uncertain")["mod_reason"] == "hedge"
    hm = table[table["semantic_group"] == "hyaline_membranes"]
    assert len(hm) == 2 and hm["bilateral_replicated"].all() and not bpn["bilateral_replicated"]
    assert one_row(table, "malaria", "diagnosis")["organ"] == "right_lung"
    assert counts["dropped"] == {
        "organ_not_allowed / steatosis / cns_post": 1,
        "positive_masked_by_uncertain / aspiration_squames / right_lung": 1,
        "uncertain_after_positive / steatosis / liver": 1,
        "unknown_text_source / autolysis / unknown_source": 1,
    }
    assert counts["nulled"] == {"steatosis / acuity / acute": 1, "steatosis / severity / plenty": 1}
    assert aggregate_findings({})[0].empty


def test_concurrence_copy():
    payload = {
        "findings": [
            {"organ": "liver", "text_source": "site_report", "condition": "steatosis",
             "severity": "mild", "extent": None, "modifiers": {}, "supporting_quote": "steatosis"},
            {"organ": "right_lung", "text_source": "site_report", "condition": "bronchopneumonia",
             "severity": "moderate", "extent": None, "modifiers": {}, "supporting_quote": "bpn"},
            {"organ": "right_lung", "text_source": "cpl_review_of_scans",
             "condition": "bronchopneumonia", "severity": "severe", "extent": None,
             "modifiers": {}, "supporting_quote": "bpn, severe"},
            {"organ": "left_lung", "text_source": "site_report", "condition": "edema",
             "severity": None, "extent": None, "modifiers": {}, "supporting_quote": "edema"},
        ],
        "negations": [{"organ": "liver", "text_source": "site_report",
                       "negation_group": "no_emh", "supporting_quote": "no EMH"}],
        "quality_flags": [
            {"organ": "liver", "text_source": "cpl_review_of_scans", "flag": "concurs_with_site",
             "supporting_quote": "Concur with site findings"},
            {"organ": "right_lung", "text_source": "cpl_review_of_scans",
             "flag": "concurs_with_site", "supporting_quote": "Concur with site findings"},
            {"organ": "liver", "text_source": "site_report", "flag": "autolysis",
             "supporting_quote": "autolysis"},
        ],
        "uncertain": [],
    }
    payload["quality_flags"].append({"organ": "liver", "text_source": "cpl_slides",
                                     "flag": "concurs_with_site", "supporting_quote": "As above"})
    assert copy_concurred_site_records(payload) == 4  # liver steatosis and no_emh, twice
    copied = [r for r in payload["findings"] + payload["negations"]
              if r.get("derived_from_concurrence")]
    assert {(r["organ"], r.get("condition") or r.get("negation_group"), r["text_source"])
            for r in copied} == {
        ("liver", "steatosis", "cpl_review_of_scans"), ("liver", "no_emh", "cpl_review_of_scans"),
        ("liver", "steatosis", "cpl_slides"), ("liver", "no_emh", "cpl_slides"),
    }
    # The CPL's own bronchopneumonia record stands; the left lung is not endorsed;
    # quality flags are never copied.
    assert len([r for r in payload["findings"] if r["text_source"] == "cpl_review_of_scans"]) == 2
    assert len(payload["quality_flags"]) == 4
    table, _ = aggregate_findings({"C": payload})
    assert table["derived_from_concurrence"].sum() == 4


def test_output_json_schema(schema, schema_v45):
    json_schema = output_json_schema(schema_v45)
    finding = json_schema["properties"]["findings"]["items"]
    assert finding["additionalProperties"] is False
    assert list(finding["properties"]) == [
        "organ", "text_source", "condition", "severity", "extent", "modifiers", "supporting_quote",
    ]
    assert set(finding["properties"]["organ"]["enum"]) == set(schema_v45.organs)
    assert "bronchopneumonia" in finding["properties"]["condition"]["enum"]
    modifiers = finding["properties"]["modifiers"]["properties"]
    assert "neutrophilic" in modifiers["inflammation_character"]["anyOf"][0]["enum"]
    assert modifiers["polarizable"]["anyOf"][0]["enum"] == ["no", "yes"]
    diagnosis = json_schema["properties"]["diagnoses"]["items"]["properties"]["diagnosis"]
    assert "malaria" in diagnosis["enum"]
    assert "diagnoses" not in output_json_schema(schema)["properties"]  # v4.4 has none
    assert json.dumps(json_schema)  # serialisable


def test_parse_reply(tmp_path):
    assert extract_findings.parse_reply('```json\n{"findings": []}\n```') == {"findings": []}
    assert extract_findings.parse_reply('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        extract_findings.parse_reply("no json here")


def test_load_records_status_and_quotes(tmp_path):
    record = {"findings": [{"organ": "liver", "text_source": "cpl_slides", "condition": "steatosis",
                            "supporting_quote": "steatosis, mild"},
                           {"negation_group": "unremarkable"}, {}],
              "negations": [{"organ": "liver", "text_source": "site_report",
                             "negation_group": "no_emh", "supporting_quote": "not in the text"}],
              "quality_flags": [], "uncertain": []}
    (tmp_path / "A.json").write_text(json.dumps(record))
    (tmp_path / "A.error.txt").write_text("retried later")
    (tmp_path / "B.error.txt").write_text("failed")
    (tmp_path / "corpus.txt").write_text("A\nB\nC\n")
    payload, counts = finalize_extraction.load_records(tmp_path)
    assert len(payload["A"]["findings"]) == 1 and len(payload["A"]["negations"]) == 2
    assert counts == {"files": 1, "moved_negations": 1, "dropped_malformed": 1}
    findings = pd.DataFrame({"champs_deid": ["A", "A"]})
    with pytest.raises(SystemExit, match="1 corpus cases"):
        finalize_extraction.case_status(tmp_path, payload, tmp_path / "corpus.txt", findings)
    status = finalize_extraction.case_status(tmp_path, payload, None, findings)
    assert status.set_index("champs_deid")["status"].to_dict() == {"A": "extracted", "B": "failed"}
    assert status.set_index("champs_deid")["n_rows"].to_dict() == {"A": 2, "B": 0}
    texts = {"A": "SITE REPORT MAJOR FINDINGS:\nLiver: no EMH.\n"
                  "CPL SLIDES:\nLiver: steatosis,  mild."}
    rate, sections = finalize_extraction.quote_checks(payload, texts)
    assert rate == round(1 / 3, 4)  # one quote is empty, one is not in the text
    assert sections["cpl_slides"]["in_named_section"] == 1
    assert sections["site_report"] == {"n": 1, "in_named_section": 0, "rate": 0.0}
    assert sections[None]["n"] == 1  # the misplaced negation names no section


def test_score_against_annotations(tmp_path):
    annotation = {
        "case_id": "A", "status": "complete",
        "annotated_cells": [["right_lung", "cpl_slides"], ["liver", "cpl_slides"]],
        "findings": [
            {"organ": "right_lung", "text_source": "cpl_slides", "condition": "bronchopneumonia",
             "severity": "extensive", "negated": False, "uncertain": False},
            {"organ": "liver", "text_source": "cpl_slides", "condition": "steatosis",
             "severity": "mild", "negated": False, "uncertain": False},
            {"organ": "liver", "text_source": "cpl_slides", "condition": "hemozoin_pigment",
             "severity": None, "negated": False, "uncertain": True},
        ],
    }
    (tmp_path / "A.json").write_text(json.dumps(annotation))
    findings = pd.DataFrame([
        {"champs_deid": "A", "organ": "right_lung", "text_source": "cpl_slides",
         "kind": "condition", "semantic_group": "bronchopneumonia", "severity": "severe"},
        {"champs_deid": "A", "organ": "right_lung", "text_source": "cpl_slides",
         "kind": "condition", "semantic_group": "hyaline_membranes", "severity": None},
        {"champs_deid": "A", "organ": "liver", "text_source": "cpl_slides",
         "kind": "uncertain", "semantic_group": "steatosis", "severity": None},
    ])
    annotations = score_against_annotations.load_annotations(tmp_path)
    extraction = score_against_annotations.extraction_verdicts(findings, {"A"})
    bpn, errors = score_against_annotations.score(annotations, extraction, "bronchopneumonia")
    assert (bpn["tp"], bpn["fp"], bpn["fn"], bpn["tn"]) == (1, 0, 0, 1)
    assert bpn["severity_agree"] == 1 and bpn["severity_n"] == 1  # extensive maps to severe
    hm, errors = score_against_annotations.score(annotations, extraction, "hyaline_membranes")
    assert hm["fp"] == 1 and errors == {"A"}
    steatosis, _ = score_against_annotations.score(annotations, extraction, "steatosis")
    assert steatosis["abstained"] == 1 and steatosis["fn"] == 0
    hemozoin, _ = score_against_annotations.score(annotations, extraction, "hemozoin_pigment")
    assert hemozoin["gold_uncertain"] == 1 and hemozoin["tn"] == 1
    assert score_against_annotations.wilson(9, 10) == (0.596, 0.982)
