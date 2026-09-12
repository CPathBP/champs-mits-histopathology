"""Per-case views of the report text: the longest text, the template, and the origin."""

import pandas as pd

from champs_pipeline.data_prep.reports import (
    ORIGIN_LABELS,
    attribute_origin,
    classify_template,
    strip_hetext_html,
)

# When a case has several report rows, the row of the highest-priority
# template, then origin, represents the case.
TEMPLATE_PRIORITY = {"HTML": 0, "plain_Hdr": 1, "DeCoDe": 2, "plain_NoHdr": 3, "empty": 4}
ORIGIN_PRIORITY = {label: i for i, label in enumerate(ORIGIN_LABELS)}
CASE_COLUMNS = ["champs_deid", "template", "origin", "has_hetext"]


def longest_text_per_case(reports):
    """Cleaned report text per case; of several rows, the longest text represents the case."""
    rows = reports[reports["champs_deid"].notna()].copy()
    rows["text"] = rows["HEText"].apply(strip_hetext_html)
    rows = rows.iloc[rows["text"].str.len().argsort(kind="stable")]
    return dict(zip(rows["champs_deid"].astype(str), rows["text"]))


def attribute_rows(reports):
    """Template and origin of every report row; rows without a case id are dropped."""
    rows = reports[reports["champs_deid"].notna()].copy()
    rows["HEText_clean"] = rows["HEText"].apply(strip_hetext_html)
    rows["has_hetext"] = rows["HEText_clean"].astype(bool)
    has_dx = zip(rows["DxAgentCode"].notna(), rows["DXSystemCode"].notna(),
                 rows["DXSyndromeCode"].notna())
    rows["template"] = [
        classify_template(raw, clean, has_dxagent=agent, has_dxsystem=system,
                          has_dxsyndrome=syndrome)
        for raw, clean, (agent, system, syndrome)
        in zip(rows["HEText"], rows["HEText_clean"], has_dx)
    ]
    rows["origin"] = rows["HEText_clean"].apply(attribute_origin)
    return rows


def build_case_attribution(reports):
    """One row per case: template, origin, and whether the case has report text.

    ``reports`` is the report table or the path of its CSV. Returns
    ``(rows, cases)``: the per-row table and the per-case table. A case
    with several rows takes the template and origin of its
    highest-priority row and has text when any row has.
    """
    if not isinstance(reports, pd.DataFrame):
        reports = pd.read_csv(reports)
    rows = attribute_rows(reports)
    rows["_template_rank"] = rows["template"].map(TEMPLATE_PRIORITY).fillna(99)
    rows["_origin_rank"] = rows["origin"].map(ORIGIN_PRIORITY).fillna(99)
    ordered = rows.sort_values(["champs_deid", "_template_rank", "_origin_rank"])
    cases = ordered.drop_duplicates("champs_deid", keep="first").set_index("champs_deid")
    cases["has_hetext"] = rows.groupby("champs_deid")["has_hetext"].any()
    cases = cases.reset_index()[CASE_COLUMNS]
    return rows, cases
