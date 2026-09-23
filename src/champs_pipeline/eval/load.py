"""Explicit checked-evidence boundary for canonical prediction artifacts."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from champs_pipeline.eval.predictions import (check_cells, record_path, sha256,
                                              validate_predictions)


@dataclass
class Predictions:
    frame: pd.DataFrame
    path: Path

    def check(self, expected=None, draft=False):
        """Check source cells, or the bound production completeness record for consumers.

        A draft check validates the table contract only. It does not require the
        inference record, so a display can render from work-in-progress predictions.
        The display then marks its provenance as a draft.
        """
        self.frame = validate_predictions(self.frame)
        if expected is not None:
            check_cells(self.frame, expected)
            return self
        if draft:
            return self
        record = json.loads(record_path(self.path).read_text())
        if (record.get("schema_version") != 1 or record.get("status") != "complete"
                or record.get("git_dirty") is not False or not record.get("git_commit")
                or record.get("artifact_sha256") != sha256(self.path)):
            raise ValueError("predictions have no matching complete clean inference record")
        counts = {str(k): int(v) for k, v in self.frame.groupby("run_id").size().items()}
        if record.get("cells_by_run") != counts or set(record.get("runs", {})) != set(counts):
            raise ValueError("prediction coverage differs from its inference record")
        return self


def predictions(path):
    """Read predictions; callers must call check() before using the frame."""
    path = Path(path)
    return Predictions(pd.read_parquet(path), path)
