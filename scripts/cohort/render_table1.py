"""Render Table 1 from the case table.

Three case sets side by side: cases with an examined report, cases with a
linked slide, and cases with a training slide. Shares are among the cases
of the set that have a row in the demographics table; the first rows
give both counts. Writes ``table1.csv`` (every level) and
``table1_compact.csv`` (binned years, the five largest cause groups).
"""

import argparse
from pathlib import Path

import pandas as pd

YEAR_BINS = [2015, 2019, 2021, 2024]
YEAR_LABELS = ["2016-2019", "2020-2021", "2022-2024"]
COHORTS = {"examined": "examined", "linked": "linked", "training": "training"}


class Table:
    """Collects the rows of a descriptive table over several case sets."""

    def __init__(self, frames):
        self.frames = frames
        self.rows = []

    def add(self, variable, level, cell):
        """One row; ``cell(name, frame)`` renders the cell of each case set."""
        row = {"variable": variable, "level": level}
        row.update({name: cell(name, frame) for name, frame in self.frames.items()})
        self.rows.append(row)

    def categorical(self, variable, column, order=None, top=None):
        """Count and share per level, an ``other`` row when ``top`` limits the levels, missing."""
        levels = order
        if levels is None:
            counts = self.frames["examined"][column].value_counts()
            levels = list(counts.head(top).index) if top else list(counts.index)

        def share(frame, mask):
            return f"{int(mask.sum()):,} ({mask.mean():.1%})"

        for level in levels:
            self.add(variable, str(level), lambda n, f, lv=level: share(f, f[column] == lv))
        if top:
            self.add(variable, "other",
                     lambda n, f: share(f, f[column].notna() & ~f[column].isin(levels)))
        self.add(variable, "missing", lambda n, f: f"{int(f[column].isna().sum()):,}")

    def continuous(self, variable, column):
        """Median and interquartile range, then the missing count."""
        def summary(name, frame):
            values = frame[column].dropna()
            if not len(values):
                return "-"
            return (f"{values.median():.1f} [{values.quantile(0.25):.1f}-"
                    f"{values.quantile(0.75):.1f}]")
        self.add(variable, "median [IQR]", summary)
        self.add(variable, "missing", lambda n, f: f"{int(f[column].isna().sum()):,}")

    def frame(self):
        return pd.DataFrame(self.rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cases = pd.read_csv(args.cohort_dir / "cohort_cases.csv", low_memory=False)
    cases["year_bin"] = pd.cut(cases["death_year"], YEAR_BINS, labels=YEAR_LABELS)
    sizes = {name: int(cases[flag].sum()) for name, flag in COHORTS.items()}
    frames = {name: cases[cases[flag] & cases["in_demographics"]] for name, flag in COHORTS.items()}

    full = Table(frames)
    full.add("cases", "in cohort", lambda n, f: f"{sizes[n]:,}")
    full.add("cases", "with a demographics row", lambda n, f: f"{len(f):,}")
    full.categorical("death category", "death_category",
                     order=["stillbirth", "neonate", "infant/child"])
    full.categorical("age group (release)", "age_group")
    full.categorical("sex", "sex")
    full.continuous("gestational age, weeks", "GA_child_wks")
    full.continuous("post-mortem interval, hours", "calc_postmortem_hrs")
    full.categorical("site", "site_iso_code")
    full.categorical("year of death", "death_year",
                     order=sorted(frames["examined"]["death_year"].dropna().unique().astype(int)))
    full.categorical("location of death", "location_of_death")
    full.categorical("underlying cause group", "underlying_cause_group", top=8)

    compact = Table(frames)
    compact.add("cases", "in cohort", lambda n, f: f"{sizes[n]:,}")
    compact.categorical("death category", "death_category",
                        order=["stillbirth", "neonate", "infant/child"])
    compact.categorical("sex", "sex")
    compact.continuous("gestational age, weeks", "GA_child_wks")
    compact.continuous("post-mortem interval, hours", "calc_postmortem_hrs")
    compact.categorical("site", "site_iso_code")
    compact.categorical("year of death", "year_bin", order=YEAR_LABELS)
    compact.categorical("underlying cause group", "underlying_cause_group", top=5)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full.frame().to_csv(args.out_dir / "table1.csv", index=False)
    compact.frame().to_csv(args.out_dir / "table1_compact.csv", index=False)
    print(compact.frame().to_string(index=False))


if __name__ == "__main__":
    main()
