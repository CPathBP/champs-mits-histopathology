"""Cohort construction, Reference A rules, the manifest join and the fold designs on toy data."""

import numpy as np
import pandas as pd
import pytest

from champs_pipeline.data_prep import cohort as co
from champs_pipeline.data_prep import folds as fo
from champs_pipeline.data_prep import reference_a as ra
from champs_pipeline.data_prep.manifest_builder import build_manifest
from champs_pipeline.data_prep.tissue import organ_of_slide, tissue_code

CASE = "C1"
LUNGS = ["right_lung", "left_lung"]
SLIDE_SOURCE = {"site_report": "SITE", "cpl_review_of_scans": "SITE",
                "cpl_slides": "CPL", "implicit_cpl": "CPL"}
# A name with an accession prefix is a central-laboratory scan; a plain specimen name a site scan.
C1_RIGHT_SITE = "M00001.043_KEAA00001"
C1_LEFT_CPL = "2019-0001-A_HE_KEAA00001_M00001.045"
C1_LIVER_CPL = "2019-0001-B_HE_KEAA00001_M00001.041"
C2_RIGHT_SITE = "M00002.043_KEAA00002"
C3_RIGHT_CPL = "2019-0003-A_HE_KEAA00003_M00003.043"
C3_LEFT_CPL = "2019-0003-B_HE_KEAA00003_M00003.045"


def record(case, organ, text_source, finding, kind="condition", **extra):
    row = {"champs_deid": case, "organ": organ, "text_source": text_source,
           "slide_source": SLIDE_SOURCE[text_source], "semantic_group": finding, "kind": kind,
           "finding_examples": finding, "severity": None, "extent": None,
           "derived_from_concurrence": False, "bilateral_replicated": False}
    row.update(extra)
    return row


@pytest.fixture
def findings():
    return pd.DataFrame([
        record(CASE, "right_lung", "cpl_review_of_scans", "bronchopneumonia", severity="mild"),
        record(CASE, "right_lung", "cpl_review_of_scans", "unremarkable", kind="negation"),
        record(CASE, "left_lung", "cpl_slides", "hyaline_membranes", kind="uncertain"),
        record(CASE, "left_lung", "cpl_slides", "diffuse_alveolar_damage"),
        record(CASE, "liver", "cpl_slides", "pigment_unspecified"),
        record(CASE, "liver", "cpl_review_of_scans", "steatosis"),
        record(CASE, "liver", "cpl_slides", "autolysis", kind="quality", severity="severe"),
        record("C2", "right_lung", "site_report", "pneumonitis"),
        record("C2", "liver", "cpl_slides", "not_performed", kind="quality"),
        record("C3", "right_lung", "cpl_slides", "inadequate_for_dx", kind="quality"),
        record("C3", "left_lung", "cpl_slides", "out_of_focus", kind="quality"),
        record("C3", "liver", "cpl_slides", "fibrin", finding_examples="fibrin lining the alveoli"),
    ])


def slide(slide_id, study, site="Kenya"):
    return {"slide_id": slide_id, "wsi_path": f"{site}/Cases/{study}/{slide_id}.svs",
            "site": site, "case_id": study, "scanner_power": 40.0}


@pytest.fixture
def inventory():
    return pd.DataFrame([
        slide(C1_RIGHT_SITE, "KEAA00001"),
        slide(C1_LEFT_CPL, "KEAA00001"),
        slide(C1_LIVER_CPL, "KEAA00001"),
        slide("2019-0001-C_GRAM_KEAA00001_M00001.043", "KEAA00001"),  # not H&E by name
        slide("M00001.047_KEAA00001", "KEAA00001"),                     # CNS
        slide("M00001.UKN_KEAA00001", "KEAA00001"),                     # no tissue code
        slide(C2_RIGHT_SITE, "KEAA00002"),                              # site report only
        slide(C3_RIGHT_CPL, "KEAA00003"),                               # inadequate
        slide(C3_LEFT_CPL, "KEAA00003"),                                # out of focus only
        slide("M00004.043_KEAA00004", "KEAA00004"),                     # no case mapping
        slide(C1_LIVER_CPL, "KEAA00009"),                               # a copy under another case
        slide("M00001.041 - 2023-01-05 10.00.00", "KEAA00001"),         # an earlier scan
        slide("M00001.041 - 2024-02-06 11.00.00", "KEAA00001"),         # and its rescan
        slide("M00001.043.GM", "KEAA00001"),                            # a Gram stain
    ])


@pytest.fixture
def index(inventory):
    rows = []
    for row in inventory.itertuples():
        for encoder, resolution in (("virchow2", "20x_224px_0px_overlap"),
                                    ("uni_v2", "20x_256px_0px_overlap")):
            if encoder == "uni_v2" and row.slide_id == C1_LIVER_CPL:
                continue
            rows.append({"slide_id": row.slide_id, "encoder": encoder, "resolution": resolution,
                         "path": f"/features/{encoder}/{row.slide_id}.h5", "timestamp": "2026",
                         "level0_magnification": 40.0, "patch_size_level0": 448.0,
                         "n_patches": 100, "wsi_path": row.wsi_path, "wsi_objective_power": 40.0,
                         "refused": None, "id_conflict": False})
    return pd.DataFrame(rows)


@pytest.fixture
def built(inventory, index, findings):
    mapping = pd.DataFrame({"study_id": ["KEAA00001", "KEAA00002", "KEAA00003"],
                            "champs_deid": [CASE, "C2", "C3"]})
    stain = pd.DataFrame({"slide_id": inventory["slide_id"], "p_non_HE": 0.01})
    encoders = {"virchow2": "20x_224px_0px_overlap", "uni_v2": "20x_256px_0px_overlap"}
    slides = co.slide_table(inventory, mapping)
    return co.build_cohort(slides, index, stain, findings, encoders, "virchow2")


def test_tissue_code_and_organ():
    assert tissue_code("2019-0001-A_HE_KEAA00001_M00001.043") == "043"
    assert organ_of_slide("M00001_045_ETAA00001") == "left_lung"
    assert tissue_code("2017-1474-C_HE_MZAA00045_M000430.46") == "046"
    assert tissue_code("M09005.0432022-07-27 09.46.09") is None
    assert organ_of_slide("M00001.UKN_KEAA00001") is None


def test_duplicate_file_keeps_the_named_case(inventory):
    mapping = pd.DataFrame({"study_id": ["KEAA00001"], "champs_deid": [CASE]})
    slides = co.slide_table(inventory, mapping).set_index("slide_id")
    assert slides.loc[C1_LIVER_CPL, "study_id"] == "KEAA00001"
    assert slides.loc[C1_LIVER_CPL, "n_files"] == 2
    assert slides.loc[C1_RIGHT_SITE, "n_files"] == 1


def test_funnel_drops_each_slide_once_with_its_stage(built):
    _, dropped, boxes = built
    assert not dropped["slide_id"].duplicated().any()
    stage = dropped.set_index("slide_id")["stage"]
    assert stage["2019-0001-C_GRAM_KEAA00001_M00001.043"] == "not_he_by_name"
    assert stage["M00001.UKN_KEAA00001"] == "no_tissue_code"
    assert stage["M00001.047_KEAA00001"] == "not_target_organ"
    assert stage["M00004.043_KEAA00004"] == "no_case_mapping"
    assert stage["M00001.043.GM"] == "not_he_by_name"
    assert stage["M00001.041 - 2023-01-05 10.00.00"] == "older_scan"
    assert stage[C2_RIGHT_SITE] == "no_cpl_authored_record"
    # A description with the inadequate flag and nothing else counts as inadequate.
    assert stage[C3_RIGHT_CPL] == "inadequate_for_dx"
    assert stage[C3_LEFT_CPL] == "quality_flags_only"
    assert boxes[0]["n"] == 13 and boxes[-1]["n"] == 4
    assert "M00001.041 - 2024-02-06 11.00.00" not in set(dropped["slide_id"])


def test_linked_and_training_flags(built):
    cohort, _, _ = built
    flags = cohort.set_index("slide_id")
    # The site report describes C2's right lung, so the slide is linked but not trainable.
    assert flags.loc[C2_RIGHT_SITE, ["linked", "training"]].tolist() == [True, False]
    assert flags.loc[C1_LIVER_CPL, ["linked", "training"]].tolist() == [True, True]
    assert cohort.loc[cohort["training"], "feature_path_virchow2"].notna().all()
    assert pd.isna(flags.loc[C1_LIVER_CPL, "feature_path_uni_v2"])
    co.check_cohort(cohort)


def test_case_reference_keeps_positive_over_uncertain(findings, schema_v45):
    reference = ra.case_reference(findings, schema_v45, LUNGS)
    status = reference.set_index(["organ", "finding"])["status"]
    assert status[("right_lung", "bronchopneumonia")] == "positive"
    assert status[("left_lung", "hyaline_membranes")] == "uncertain"
    assert ("right_lung", "unremarkable") not in status.index


def test_slide_reference_mask_reasons(built, findings, schema_v45):
    cohort, _, _ = built
    lungs = cohort[cohort["training"] & (cohort["organ_group"] == "lung")]
    reference = ra.case_reference(findings, schema_v45, LUNGS)
    labels = ["bronchopneumonia", "hyaline_membranes", "pneumonitis", "fibrin"]
    table = ra.slide_reference(lungs, reference, findings, labels, LUNGS)
    table = table.set_index(["organ", "finding", "variant"])
    assert table.loc[("right_lung", "bronchopneumonia", "raw"), "label"] == 1.0
    assert table.loc[("right_lung", "bronchopneumonia", "raw"), "severity"] == "mild"
    assert table.loc[("left_lung", "hyaline_membranes", "raw"), "mask_reason"] == "uncertain"
    # The right lung is silent on hyaline membranes; the left lung hedges it, so the case blocks.
    assert table.loc[("right_lung", "hyaline_membranes", "raw"), "label"] == 0.0
    assert table.loc[("right_lung", "hyaline_membranes", "elig"), "mask_reason"] == "eligibility"
    assert table.loc[("right_lung", "pneumonitis", "elig"), "mask_reason"] is None
    assert table.loc[("right_lung", "fibrin", "elig"), "mask_reason"] is None
    assert table["mask"].eq(table["label"].notna().astype(float)).all()


def test_related_and_quality_units(findings):
    assert (CASE, "left_lung", "CPL") in ra.related_units(findings, "diffuse_alveolar_damage")
    assert ("C3", "liver", "CPL") in ra.related_units(findings, "fibrin_lining")
    assert (CASE, "liver", "CPL") in ra.related_units(findings, "pigment_unspecified")
    # The findings table stores modifiers as nullable text; an unstated value still masks.
    findings["mod_polarizable"] = pd.array([pd.NA] * len(findings), dtype="string")
    assert (CASE, "liver", "CPL") in ra.related_units(findings, "pigment_unspecified")
    findings.loc[findings["semantic_group"] == "pigment_unspecified", "mod_polarizable"] = "no"
    assert (CASE, "liver", "CPL") not in ra.related_units(findings, "pigment_unspecified")
    assert ra.quality_units(findings) == {(CASE, "liver", "CPL"), ("C3", "left_lung", "CPL")}


def test_finding_supply_counts_cases_with_a_positive_training_slide():
    import render_finding_burden

    unit = {"finding": "steatosis", "status": "positive"}
    reference = pd.DataFrame([
        {"champs_deid": "A", "organ": "liver", "slide_source": "CPL", **unit},
        {"champs_deid": "B", "organ": "liver", "slide_source": "SITE", **unit},
    ])
    # Case B has a training slide, but of the lung; its positive liver slide is linked only.
    cohort = pd.DataFrame([
        {"champs_deid": "A", "organ": "liver", "slide_source": "CPL", "linked": True,
         "training": True},
        {"champs_deid": "B", "organ": "right_lung", "slide_source": "SITE", "linked": True,
         "training": True},
        {"champs_deid": "B", "organ": "liver", "slide_source": "SITE", "linked": True,
         "training": False},
    ])
    cases = pd.DataFrame({"champs_deid": ["A", "B"], "examined": [True, True]})
    groups = {"liver": "liver", "right_lung": "lung"}
    supply = render_finding_burden.finding_supply(reference, cohort, cases, groups)
    row = supply.set_index("finding").loc["steatosis"]
    assert row["positive_cases"] == 2 and row["units_with_linked_slide"] == 2
    assert row["cases_with_positive_training_slide"] == 1


def test_manifest_join(built, findings, schema_v45):
    cohort, _, _ = built
    lungs = cohort[cohort["training"] & (cohort["organ_group"] == "lung")]
    reference = ra.case_reference(findings, schema_v45, LUNGS)
    config = {"organ": "lung", "organs": LUNGS, "labels": ["bronchopneumonia", "hyaline_membranes"],
              "negation_flags": ["unremarkable"], "quality_flags": ["autolysis"],
              "modifiers": ["inflammation_character"]}
    rows = ra.slide_reference(lungs, reference, findings, config["labels"], LUNGS)
    cases = pd.DataFrame({"champs_deid": [CASE], "template": ["HTML"], "origin": ["EXPLICIT"]})
    lance = {"virchow2": pd.DataFrame({"slide_id": lungs["slide_id"], "lance_dataset_path": "x",
                                       "lance_row_idx": range(len(lungs))})}
    manifest = build_manifest(lungs, rows[rows["variant"] == "elig"], cases, findings, schema_v45,
                              config, {"virchow2": "", "uni_v2": ""}, "virchow2", lance)
    row = manifest.set_index("organ").loc["right_lung"]
    assert row["label"] == "bronchopneumonia" and row["label_bronchopneumonia"] == 1.0
    assert row["severity_class_bronchopneumonia"] == 2 and row["neg_unremarkable"] == 1
    assert np.isnan(row["label_hyaline_membranes"]) and row["mask_hyaline_membranes"] == 0.0
    position = list(lungs["slide_id"]).index(C1_RIGHT_SITE)
    assert row["lance_row_idx"] == position and row["template"] == "HTML"
    assert list(manifest.columns[:4]) == ["slide_id", "case_id", "feature_path", "label"]


def toy_manifest(n_cases=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cases):
        site = ["A", "B", "C"][i % 3]
        for k in range(rng.integers(1, 3)):
            rows.append({"slide_id": f"s{i}_{k}", "case_id": f"c{i}", "location": site,
                         "label_x": float(i % 4 == 0), "mask_x": 1.0,
                         "label_y": float(i % 5 == 0), "mask_y": 1.0})
    return pd.DataFrame(rows)


def test_fivefold_and_site_designs_hold_out_every_case_once():
    manifest = toy_manifest()
    labels = ["x", "y"]
    case_map = fo.fivefold_case_map(manifest, labels, seed=42)
    splits = fo.fivefold_splits(case_map)
    fo.check_test_coverage(splits)
    assert set(case_map["microfold"]) == set(range(10))
    sites = fo.site_splits(manifest, labels, seed=42)
    fo.check_test_coverage(sites)
    assert sites.groupby("fold")["site"].nunique().eq(1).all()
    fold = fo.fold_manifest(manifest, sites, 0)
    assert set(fold.loc[fold["split"] == "test", "location"]) == {"A"}
    assert fold.groupby("case_id")["split"].nunique().eq(1).all()


def test_gate_flags_missing_classes():
    manifest = toy_manifest()
    splits = fo.fivefold_splits(fo.fivefold_case_map(manifest, ["x", "y"], seed=42))
    folds = {k: fo.fold_manifest(manifest, splits, k) for k in range(5)}
    table = fo.gate_table(folds, ["x", "y"], core_labels=["x"], site_design=False)
    assert not table["gate_status"].str.startswith("FAIL").any()
    folds[0].loc[folds[0]["split"] == "train", "label_x"] = 0.0
    table = fo.gate_table(folds, ["x"], core_labels=["x"], site_design=False)
    assert table.loc[table["fold"] == 0, "gate_status"].iloc[0] == "FAIL_TRAIN_CLASS"
