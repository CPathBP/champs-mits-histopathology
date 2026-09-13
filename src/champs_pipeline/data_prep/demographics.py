"""Case demographics from the release's basic-demographics table.

Age at death in months is derived from the dates of birth and death and,
when those are missing or inconsistent, from the age components. The
release's ``age_group`` (stillbirth, first day, early and late neonate,
infant, child) is complete and carries the age structure the findings
depend on; the manifest carries it one-hot together with a scaled
gestational age, so an age-conditioned model can read them as tabular
features.
"""

import pandas as pd

AGE_GROUPS = {
    "Stillbirth": "stillbirth",
    "Death in the first 24 hours": "death24h",
    "Early Neonate (1 to 6 days)": "early_neonate",
    "Late Neonate (7 to 27 days)": "late_neonate",
    "Infant (28 days to less than 12 months)": "infant",
    "Child (12 months to less than 60 Months)": "child",
}
DEATH_CATEGORY = {
    "Stillbirth": "stillbirth",
    "Death in the first 24 hours": "neonate",
    "Early Neonate (1 to 6 days)": "neonate",
    "Late Neonate (7 to 27 days)": "neonate",
    "Infant (28 days to less than 12 months)": "infant/child",
    "Child (12 months to less than 60 Months)": "infant/child",
}
DAYS_PER_MONTH = 30.4375
KEEP = ["champs_deid", "site_iso_code", "sex", "age_group", "age_group_subcat", "GA_child_wks",
        "calc_postmortem_hrs", "location_of_death"]


def age_in_months(table):
    """Age at death in months: from the dates when consistent, else from the components."""
    birth = pd.to_datetime(table["date_of_birth"], errors="coerce")
    death = pd.to_datetime(table["date_of_death"], errors="coerce")
    by_dates = (death - birth).dt.days / DAYS_PER_MONTH
    parts = {k: pd.to_numeric(table[k], errors="coerce").fillna(0.0)
             for k in ("age_years", "age_months", "age_days", "age_hours")}
    by_parts = (parts["age_years"] * 12.0 + parts["age_months"]
                + parts["age_days"] / DAYS_PER_MONTH + parts["age_hours"] / (24.0 * DAYS_PER_MONTH))
    age = by_dates.where(by_dates.notna() & (by_dates >= 0), by_parts)
    return age.where(age >= 0)


def load_demographics(path):
    """One row per case with the kept columns, the derived age and the encoded age features."""
    table = pd.read_csv(path, low_memory=False).drop_duplicates("champs_deid")
    out = table[KEEP].copy()
    out["champs_deid"] = out["champs_deid"].astype(str)
    out["death_year"] = pd.to_datetime(table["date_of_death"], errors="coerce").dt.year
    out["death_category"] = table["age_group"].map(DEATH_CATEGORY)
    out["age_months_total"] = age_in_months(table)
    out["age_missing"] = out["age_months_total"].isna().astype(int)
    for value, suffix in AGE_GROUPS.items():
        out[f"age_oh_{suffix}"] = (table["age_group"] == value).astype(int)
    gestational = pd.to_numeric(table["GA_child_wks"], errors="coerce")
    out["ga_missing"] = gestational.isna().astype(int)
    out["ga_norm"] = (gestational / 40.0).fillna(0.0)
    return out.reset_index(drop=True)
