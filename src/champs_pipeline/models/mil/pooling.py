"""Mean pooling: the low-capacity reference aggregator."""

from torch import nn


class MeanPool(nn.Module):
    """Project every tile, average the bag, classify the average."""

    def __init__(self, input_dim, n_classes, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def forward(self, bag):
        """``bag`` is (tiles, input_dim); returns ``{"logits": (1, n_classes)}``."""
        pooled = self.projection(bag).mean(dim=0, keepdim=True)
        return {"logits": self.classifier(pooled)}
