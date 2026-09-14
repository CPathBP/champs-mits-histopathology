"""Slides as bags of tile features, read from the feature store.

A fold manifest holds one row per training slide with its split, its
labels (``label_<finding>``: 1, 0, or empty), its keep masks
(``mask_<finding>``: 0 drops the cell from the loss and the metrics) and,
per encoder, the store and row of its features. The data module reads one
encoder's columns, drops the slides without features for it, and serves
the three splits as bags of variable length. It refuses a manifest whose
labels and masks contradict each other, an encoder that lacks features for
many slides, and a training or validation split without both classes of a
finding.
"""

import lance
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from champs_pipeline.datasets.wsi.learning_curve import subsample_train_by_fraction

SPLITS = ("train", "val", "test")


def collate(batch):
    """Bags stay a list because their lengths differ; labels and masks stack."""
    return {
        "features": [item["features"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
        "label_mask": torch.stack([item["label_mask"] for item in batch]),
        "slide_id": [item["slide_id"] for item in batch],
    }


def store_columns(encoder):
    """The manifest columns that locate a slide's features for one encoder."""
    return f"lance_dataset_path_{encoder}", f"lance_row_idx_{encoder}"


def read_bag(store, row, slide_id):
    """The (tiles, dim) feature array of one row of an open store, checked against the slide."""
    table = store.take([int(row)], columns=["slide_id", "features", "embed_dim"])
    stored = table["slide_id"][0].as_py()
    if stored != slide_id:
        raise ValueError(f"store row {row} holds {stored!r}, not {slide_id!r}")
    flat = table["features"][0].values.to_numpy(zero_copy_only=False)
    return flat.astype(np.float32, copy=False).reshape(-1, int(table["embed_dim"][0].as_py()))


def remap_store_paths(frame, column, stores_dir):
    """Point the store column at ``stores_dir``, keeping each store's directory name."""
    if stores_dir is None:
        return frame
    frame = frame.copy()
    present = frame[column].notna()
    names = frame.loc[present, column].map(lambda path: str(path).rstrip("/").split("/")[-1])
    frame.loc[present, column] = [f"{str(stores_dir).rstrip('/')}/{name}" for name in names]
    return frame


def check_label_cells(frame, labels, mask_prefix, source):
    """Refuse label and mask values that contradict the cohort stage's rules."""
    for label in labels:
        value, mask = frame[f"label_{label}"], frame[f"{mask_prefix}{label}"]
        problems = {
            "an empty mask": mask.isna(),
            "a mask other than 0 or 1": mask.notna() & ~mask.isin([0, 1]),
            "a label other than 0, 1 or empty": value.notna() & ~value.isin([0, 1]),
            "an empty label that is kept": value.isna() & mask.eq(1),
            "a masked positive": value.eq(1) & mask.eq(0),
        }
        for problem, cells in problems.items():
            if cells.any():
                raise ValueError(f"{source}: {int(cells.sum())} cells of {label} have {problem}")


class SlideBagDataset(Dataset):
    """The slides of one split: features from the store, labels and masks from the manifest."""

    def __init__(self, frame, labels, encoder, mask_prefix="mask_"):
        self.frame = frame.reset_index(drop=True)
        self.path_col, self.row_col = store_columns(encoder)
        targets = self.frame[[f"label_{label}" for label in labels]].fillna(0.0)
        masks = self.frame[[f"{mask_prefix}{label}" for label in labels]]
        self.targets = torch.tensor(targets.to_numpy(dtype=np.float32))
        self.masks = torch.tensor(masks.to_numpy(dtype=np.float32))
        self.slide_ids = self.frame["slide_id"].astype(str).tolist()
        self._stores = {}

    def __len__(self):
        return len(self.frame)

    def _store(self, path):
        # Opened lazily so that every loader worker holds its own handle.
        if path not in self._stores:
            self._stores[path] = lance.dataset(path)
        return self._stores[path]

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        bag = read_bag(self._store(row[self.path_col]), row[self.row_col], self.slide_ids[index])
        return {"features": torch.from_numpy(bag), "label": self.targets[index],
                "label_mask": self.masks[index], "slide_id": self.slide_ids[index]}


def class_counts(dataset):
    """Kept positives and kept negatives per label."""
    positives = (dataset.targets * dataset.masks).sum(0)
    return positives, dataset.masks.sum(0) - positives


def split_counts(dataset, labels, dropped):
    """Slides, cases, and per label the kept positives, kept negatives and masked cells."""
    positives, negatives = class_counts(dataset)
    per_label = {label: {"pos": int(positives[i]), "neg": int(negatives[i]),
                         "masked": int(len(dataset) - dataset.masks[:, i].sum())}
                 for i, label in enumerate(labels)}
    return {"slides": len(dataset), "cases": int(dataset.frame["case_id"].nunique()),
            "dropped_no_features": dropped, "labels": per_label}


class SlideBagDataModule(pl.LightningDataModule):
    """One fold manifest, one encoder, three splits of bags."""

    def __init__(self, fold_csv, labels, encoder, batch_size=4, num_workers=4,
                 train_fraction=1.0, train_fraction_seed=42, stores_dir=None,
                 mask_prefix="mask_", max_dropped_share=0.05):
        super().__init__()
        self.fold_csv = str(fold_csv)
        self.labels = list(labels)
        self.encoder = encoder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = float(train_fraction)
        self.train_fraction_seed = train_fraction_seed
        self.stores_dir = stores_dir
        self.mask_prefix = mask_prefix
        self.max_dropped_share = max_dropped_share
        self.datasets = {}
        self.dropped = {}
        self.embed_dim = None

    def setup(self, stage=None):
        if self.datasets:
            return
        frame = self._read_fold()
        path_col, row_col = store_columns(self.encoder)
        usable = frame[path_col].notna() & frame[row_col].notna()
        self._check_dropped(frame, usable)
        frame = frame[usable]
        self.embed_dim = int(lance.dataset(frame[path_col].iloc[0])
                             .take([0], columns=["embed_dim"])["embed_dim"][0].as_py())
        parts = {split: frame[frame["split"] == split] for split in SPLITS}
        if self.train_fraction < 1.0:
            parts["train"] = subsample_train_by_fraction(
                parts["train"], self.train_fraction, labels=self.labels,
                seed=self.train_fraction_seed)
        self.datasets = {split: SlideBagDataset(part, self.labels, self.encoder,
                                                self.mask_prefix)
                         for split, part in parts.items()}
        self._check_classes()

    def _read_fold(self):
        """The fold manifest with its columns, split values and label cells checked."""
        frame = pd.read_csv(self.fold_csv, low_memory=False)
        path_col, row_col = store_columns(self.encoder)
        needed = ["slide_id", "case_id", "split", path_col, row_col]
        needed += [f"label_{label}" for label in self.labels]
        needed += [f"{self.mask_prefix}{label}" for label in self.labels]
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            raise KeyError(f"{self.fold_csv} lacks {missing}")
        present = set(frame["split"])
        if present != set(SPLITS):
            raise ValueError(f"{self.fold_csv}: split values {sorted(present)}, "
                             f"expected {list(SPLITS)}")
        check_label_cells(frame, self.labels, self.mask_prefix, self.fold_csv)
        return remap_store_paths(frame, path_col, self.stores_dir)

    def _check_dropped(self, frame, usable):
        """Count the slides without features and refuse a share above the limit."""
        self.dropped, shares = {}, {}
        for split in SPLITS:
            in_split = frame["split"] == split
            self.dropped[split] = int((in_split & ~usable).sum())
            shares[split] = self.dropped[split] / int(in_split.sum())
        too_many = {split: round(share, 3) for split, share in shares.items()
                    if share > self.max_dropped_share}
        if too_many:
            raise ValueError(f"{self.encoder} lacks features for these shares of the slides of "
                             f"{self.fold_csv}: {too_many}; the limit is {self.max_dropped_share}")

    def _check_classes(self):
        """Refuse a training or validation split without a kept positive and negative per label."""
        for split in ("train", "val"):
            positives, negatives = class_counts(self.datasets[split])
            short = [label for i, label in enumerate(self.labels)
                     if positives[i] < 1 or negatives[i] < 1]
            if short:
                raise ValueError(f"{self.fold_csv}: the {split} split lacks a kept positive or "
                                 f"negative for {short}")

    def counts(self):
        return {split: split_counts(dataset, self.labels, self.dropped.get(split, 0))
                for split, dataset in self.datasets.items()}

    def pos_weight(self, clip_max=100.0):
        """Per label, kept negatives over kept positives of the training split, in [1, clip_max]."""
        positives, negatives = class_counts(self.datasets["train"])
        return (negatives.clamp(min=1.0) / positives.clamp(min=1.0)).clamp(1.0, clip_max)

    def _loader(self, split, shuffle):
        # Workers start from a fresh server process: a forked worker inherits the store
        # runtime and CUDA state of the main process and crashes on its first read.
        context = "forkserver" if self.num_workers > 0 else None
        return DataLoader(self.datasets[split], batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, collate_fn=collate, pin_memory=True,
                          persistent_workers=self.num_workers > 0,
                          multiprocessing_context=context)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)
