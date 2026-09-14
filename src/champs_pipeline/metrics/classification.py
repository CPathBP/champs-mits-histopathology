"""Multi-label metrics over the kept cells of an epoch.

Average precision, area under the ROC curve and F1 at 0.5, per label and
macro-averaged. A cell whose mask is 0 is set to the ignore index and
enters no metric.
"""

import torch
from torch import nn
from torchmetrics.classification import (
    MultilabelAUROC, MultilabelAveragePrecision, MultilabelF1Score,
)

IGNORE = -1


class MultilabelMetrics(nn.Module):
    """Per-label and macro AP, AUROC and F1 with a keep mask."""

    def __init__(self, num_labels, threshold=0.5):
        super().__init__()
        self.ap = MultilabelAveragePrecision(num_labels=num_labels, average=None,
                                             ignore_index=IGNORE, sync_on_compute=False)
        self.auroc = MultilabelAUROC(num_labels=num_labels, average=None, ignore_index=IGNORE,
                                     sync_on_compute=False)
        self.f1 = MultilabelF1Score(num_labels=num_labels, threshold=threshold, average=None,
                                    ignore_index=IGNORE, sync_on_compute=False)

    def update(self, probabilities, targets, mask):
        targets = targets.long().clone()
        targets[mask == 0] = IGNORE
        for metric in (self.ap, self.auroc, self.f1):
            metric.update(probabilities, targets)

    def compute(self):
        """``ap``, ``auroc`` and ``f1`` per label, each a vector over the labels."""
        return {"ap": self.ap.compute(), "auroc": self.auroc.compute(), "f1": self.f1.compute()}

    def reset(self):
        for metric in (self.ap, self.auroc, self.f1):
            metric.reset()


def macro(values, indices=None):
    """The mean over all labels, or over the labels at ``indices``."""
    if indices is not None:
        values = values[torch.as_tensor(indices, device=values.device)]
    return values.float().mean()
