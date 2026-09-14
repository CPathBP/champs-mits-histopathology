"""The six aggregators, by the name the training matrix uses.

Each takes ``input_dim`` and ``n_classes`` plus its architecture keywords,
consumes one bag of tile features (tiles, input_dim), and returns a dict
with ``logits`` of shape (1, n_classes).
"""

from champs_pipeline.models.mil.abmil import ABMIL
from champs_pipeline.models.mil.acmil import ACMIL
from champs_pipeline.models.mil.clam import CLAM_MB
from champs_pipeline.models.mil.pooling import MeanPool
from champs_pipeline.models.mil.transmil import TransMIL


def _mammil(**kwargs):
    from champs_pipeline.models.mil.mammil import MamMIL

    return MamMIL(**kwargs)


AGGREGATORS = {
    "meanpool": MeanPool,
    "abmil": ABMIL,
    "clam_mb": CLAM_MB,
    "acmil": ACMIL,
    "transmil": TransMIL,
    "mammil": _mammil,
}

__all__ = ["AGGREGATORS", "ABMIL", "ACMIL", "CLAM_MB", "MeanPool", "TransMIL"]
