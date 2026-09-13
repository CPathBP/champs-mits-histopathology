"""The panel-assigned cause of death (DeCoDe) per case, from the release table."""

import pandas as pd

COLUMNS = ["champs_deid", "UC_champs_group_desc", "IC_champs_group_desc"]


def load_decode_results(path):
    """One row per case: the underlying and the immediate cause group."""
    table = pd.read_csv(path, usecols=COLUMNS, low_memory=False).drop_duplicates("champs_deid")
    table["champs_deid"] = table["champs_deid"].astype(str)
    return table.rename(columns={"UC_champs_group_desc": "underlying_cause_group",
                                 "IC_champs_group_desc": "immediate_cause_group"})
