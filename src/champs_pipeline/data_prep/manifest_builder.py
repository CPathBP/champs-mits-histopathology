"""The training manifest: the cohort joined to the slide reference and the case table.

One file per organ group and label variant, one row per training slide.
No decision is taken here; every label, mask and flag comes from the
slide reference and the findings table, every path from the cohort table.
"""

import pandas as pd

from champs_pipeline.data_prep.cohort import UNIT, label_records
from champs_pipeline.data_prep.reference_a import qualifier_class
from champs_pipeline.data_prep.tissue import ORGAN_NAME

LEADING = ["slide_id", "case_id", "feature_path", "label", "wsi_path", "tissue_name",
           "location", "organ", "slide_source", "tissue_code", "template", "origin"]
CASE_COLUMNS = ["template", "origin", "sex", "age_group", "age_group_subcat", "GA_child_wks",
                "age_months_total", "age_missing", "age_oh_stillbirth", "age_oh_death24h",
                "age_oh_early_neonate", "age_oh_late_neonate", "age_oh_infant", "age_oh_child",
                "ga_norm", "ga_missing"]


def flag_columns(slides, findings, negation_flags, quality_flags):
    """Per slide: 0/1 for each negation and quality flag of its unit in the label sources."""
    rows = label_records(findings)
    rows = rows[rows["kind"].isin(["negation", "quality"])]
    fired = rows.groupby(UNIT)["semantic_group"].agg(set)
    units = pd.MultiIndex.from_frame(slides[UNIT])
    groups = fired.reindex(units).map(lambda s: s if isinstance(s, set) else set())
    out = pd.DataFrame(index=slides.index)
    for flag in negation_flags:
        out[f"neg_{flag}"] = [int(flag in g) for g in groups]
    for flag in quality_flags:
        out[f"q_{flag}"] = [int(flag in g) for g in groups]
    return out


def label_columns(slides, reference, schema, labels, modifiers):
    """Per slide: label, mask, severity, extent and modifier columns for each finding."""
    wide = reference.set_index(["slide_id", "finding"])
    out = pd.DataFrame(index=slides.index)
    for finding in labels:
        rows = wide.xs(finding, level="finding").reindex(slides["slide_id"])
        rows.index = slides.index
        out[f"label_{finding}"] = rows["label"].astype(float)
        out[f"mask_{finding}"] = rows["mask"].astype(float)
        condition = schema.spec(finding)
        for axis in ("severity", "extent"):
            out[f"{axis}_{finding}"] = rows[axis]
            ladder = getattr(condition, f"{axis}_values")
            out[f"{axis}_class_{finding}"] = [
                qualifier_class(label, value, ladder)
                for label, value in zip(rows["label"], rows[axis])
            ]
        for modifier in modifiers:
            if modifier not in condition.modifiers:
                continue
            column = f"mod_{modifier}"
            values = rows[column] if column in rows.columns else pd.Series(None, index=rows.index)
            out[f"mod_{modifier}_{finding}"] = values
            out[f"mod_{modifier}_class_{finding}"] = [
                qualifier_class(label, value, condition.modifiers[modifier])
                for label, value in zip(rows["label"], values)
            ]
    positives = [f"label_{f}" for f in labels]
    out["label"] = [",".join(f for f in labels if row[f"label_{f}"] == 1.0)
                    for _, row in out[positives].iterrows()]
    return out


def build_manifest(slides, reference, cases, findings, schema, config, encoders, cohort_encoder,
                   lance_indexes):
    """The manifest of one organ group and variant.

    ``slides`` are the training slides of the organ group; ``reference``
    the slide reference rows of the variant; ``cases`` the case table;
    ``lance_indexes`` maps an encoder to its row index (or None).
    """
    slides = slides.reset_index(drop=True)
    out = slides[["slide_id", "champs_deid", "wsi_path", "site", "organ", "slide_source",
                  "tissue_code"]].rename(columns={"champs_deid": "case_id", "site": "location"})
    out["feature_path"] = slides[f"feature_path_{cohort_encoder}"]
    out["tissue_name"] = out["organ"].map(ORGAN_NAME)
    out = out.join(label_columns(slides, reference, schema, config["labels"], config["modifiers"]))
    out = out.join(flag_columns(slides, findings, config["negation_flags"],
                                config["quality_flags"]))
    for encoder in encoders:
        out[f"feature_path_{encoder}"] = slides[f"feature_path_{encoder}"]
        index = lance_indexes.get(encoder)
        if index is None:
            continue
        rows = index.set_index("slide_id").reindex(out["slide_id"])
        out[f"lance_dataset_path_{encoder}"] = rows["lance_dataset_path"].to_numpy()
        out[f"lance_row_idx_{encoder}"] = rows["lance_row_idx"].to_numpy()
    if cohort_encoder in lance_indexes:
        out["lance_dataset_path"] = out[f"lance_dataset_path_{cohort_encoder}"]
        out["lance_row_idx"] = out[f"lance_row_idx_{cohort_encoder}"].astype("Int64")
    case_columns = [c for c in CASE_COLUMNS if c in cases.columns]
    case_table = cases[["champs_deid"] + case_columns].rename(columns={"champs_deid": "case_id"})
    out = out.merge(case_table, on="case_id", how="left")
    ordered = [c for c in LEADING if c in out.columns]
    return out[ordered + [c for c in out.columns if c not in ordered]]
