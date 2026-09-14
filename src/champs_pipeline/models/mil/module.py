"""The training module shared by the six aggregators.

One bag per slide, several bags per optimizer step. The loss is binary
cross-entropy per finding, averaged over the kept cells of the batch,
with a positive weight per finding (kept negatives over kept positives of
the training split). The weight is state of the run, not of the model, so
checkpoints hold only the model and load strictly. Tile dropout hides a
share of every bag while training. AdamW with a linear warm-up into a
cosine schedule. The
checkpoint metric is the macro average precision over the selection
findings on the validation split.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from champs_pipeline.metrics import MultilabelMetrics, macro
from champs_pipeline.models.mil import AGGREGATORS


def masked_mean(values, mask):
    """The mean of ``values`` over the cells whose mask is 1."""
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def drop_tiles(bag, share):
    """A random subset of the tiles, in their original order."""
    n_tiles = bag.shape[0]
    keep = max(1, int(n_tiles * (1.0 - share)))
    if n_tiles <= 1 or keep >= n_tiles:
        return bag
    indices = torch.randperm(n_tiles, device=bag.device)[:keep].sort().values
    return bag[indices]


class MILModule(pl.LightningModule):
    """One aggregator, its loss, metrics and optimizer."""

    def __init__(self, aggregator, input_dim, labels, selection_labels, architecture=None,
                 learning_rate=5e-5, weight_decay=5e-4, warmup_epochs=2, instance_dropout=0.25,
                 pos_weight_max=100.0):
        super().__init__()
        self.save_hyperparameters()
        self.labels = list(labels)
        self.selection = [self.labels.index(label) for label in selection_labels]
        self.model = AGGREGATORS[aggregator](input_dim=input_dim, n_classes=len(self.labels),
                                             **(architecture or {}))
        self.register_buffer("pos_weight", torch.ones(len(self.labels)), persistent=False)
        self.train_metrics = MultilabelMetrics(len(self.labels))
        self.val_metrics = MultilabelMetrics(len(self.labels))
        self.test_metrics = MultilabelMetrics(len(self.labels))

    def setup(self, stage=None):
        # Every stage with a training split gets the weights, so a validation or test pass
        # reports the loss the run trained with.
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and "train" in getattr(datamodule, "datasets", {}):
            self.pos_weight = datamodule.pos_weight(self.hparams.pos_weight_max).to(self.device)

    def forward(self, bags):
        """Logits (bags, findings) and the summed extra loss of the aggregator, if any."""
        outputs = [self.model(bag) for bag in bags]
        logits = torch.cat([out["logits"] for out in outputs], dim=0)
        extras = [out["extra_loss"] for out in outputs if "extra_loss" in out]
        return logits, (torch.stack(extras).mean() if extras else None)

    def _step(self, batch, split):
        bags = batch["features"]
        if split == "train" and self.hparams.instance_dropout > 0:
            bags = [drop_tiles(bag, self.hparams.instance_dropout) for bag in bags]
        logits, extra = self(bags)
        labels, mask = batch["label"], batch["label_mask"]
        cells = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight,
                                                   reduction="none")
        loss = masked_mean(cells, mask)
        if extra is not None:
            loss = loss + extra
        self.log(f"{split}_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(bags))
        getattr(self, f"{split}_metrics").update(torch.sigmoid(logits.float()), labels, mask)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def _epoch_end(self, split):
        computed = getattr(self, f"{split}_metrics").compute()
        for name, values in computed.items():
            self.log(f"{split}_{name}_macro", macro(values))
            for i, label in enumerate(self.labels):
                self.log(f"{split}_{name}_{label}", values[i].float())
        self.log(f"{split}_ap_core_macro", macro(computed["ap"], self.selection))
        getattr(self, f"{split}_metrics").reset()

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        warmup = self.hparams.warmup_epochs
        # At least one cosine epoch, so a run shorter than the warm-up still has a schedule.
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.trainer.max_epochs - warmup), eta_min=1e-6)
        if warmup > 0:
            linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                                       end_factor=1.0, total_iters=warmup)
            cosine = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear, cosine],
                                                           milestones=[warmup])
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": cosine, "interval": "epoch", "frequency": 1}}
