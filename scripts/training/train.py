"""Train one run of the training matrix.

Reads the run's row from the expanded matrix, its fold manifest, the label
configuration of its organ, the model and trainer configurations, and the
feature store of its encoder. Writes into ``<runs-dir>/<run_id>/``:
``run.json`` (identity, code state, input hashes, status, selected
checkpoint), ``config.json`` (every setting the run used),
``label_mapping.json`` (the head order), ``data_counts.json`` (slides,
cases and kept classes per split), ``metrics.csv`` and ``hparams.yaml``
(per epoch and per hyperparameter, from Lightning), and ``checkpoints/``.

The checkpoint of the run is the one with the highest validation macro
average precision over the selection findings; it is resolved once, here,
and recorded in ``run.json``.

A complete run is kept when its matrix row and input hashes are unchanged
and refused otherwise. An attempt that did not complete, and whose job is
no longer queued, is moved under ``<runs-dir>/.attempts/`` and the run is
trained again; this is what a requeued or relaunched job does.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from champs_pipeline.datasets.wsi.dataset import SlideBagDataModule
from champs_pipeline.eval import run_record
from champs_pipeline.models.mil.module import MILModule
from champs_pipeline.training.matrix import fold_dir

REPO = Path(__file__).resolve().parents[2]


def matrix_row(matrix_csv, run_id):
    """The row of one run, as JSON values; refuses an unknown or duplicated id."""
    rows = pd.read_csv(matrix_csv)
    row = rows[rows["run_id"] == run_id]
    if len(row) != 1:
        raise KeyError(f"{run_id} is in {matrix_csv} {len(row)} times")
    return json.loads(row.iloc[0].to_json())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def set_aside(run_dir):
    """Move an unfinished attempt to ``.attempts/<run_id>/<time>`` and return where it went."""
    target = run_dir.parent / ".attempts" / run_dir.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    target.parent.mkdir(parents=True, exist_ok=True)
    run_dir.rename(target)
    return target


def clear_previous(run_dir, row, inputs):
    """``True`` when the run is already complete; otherwise the directory is free afterwards."""
    if not run_dir.exists():
        return False
    previous = run_record.read(run_dir) or {"status": "missing"}
    if previous["status"] == "complete":
        if previous.get("matrix_row") != row or previous.get("inputs") != inputs:
            raise SystemExit(f"{run_dir} is complete but its matrix row or inputs differ from "
                             "the current ones; move it away to train the run again")
        return True
    if run_record.is_live(previous):
        raise SystemExit(f"{run_dir} belongs to job {previous['slurm_job_id']}, still queued")
    print(f"previous attempt ({previous['status']}) moved to {set_aside(run_dir)}")
    return False


def build_trainer(run_dir, trainer_cfg):
    """The trainer with the checkpoint, early-stopping and learning-rate callbacks."""
    monitor = trainer_cfg["monitor"]
    checkpoint = ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="{epoch:02d}-{%s:.4f}"
                                 % monitor, monitor=monitor, mode="max",
                                 save_top_k=trainer_cfg["save_top_k"], save_last=True)
    early_stopping = EarlyStopping(monitor=monitor, mode="max", patience=trainer_cfg["patience"],
                                   strict=True)
    trainer = pl.Trainer(
        accelerator="auto", devices=1, precision=trainer_cfg["precision"],
        max_epochs=trainer_cfg["max_epochs"], gradient_clip_val=trainer_cfg["gradient_clip_val"],
        log_every_n_steps=trainer_cfg["log_every_n_steps"], enable_progress_bar=False,
        callbacks=[checkpoint, early_stopping, LearningRateMonitor(logging_interval="epoch")],
        logger=CSVLogger(save_dir=run_dir, name="", version=""))
    return trainer, checkpoint, early_stopping


def device_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True, help="The expanded training matrix.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--folds-dir", type=Path, required=True,
                    help="The fold_csvs directory of the cohort stage.")
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--stores-dir", help="Where the feature stores are, if they moved.")
    ap.add_argument("--configs", type=Path, default=REPO / "configs")
    ap.add_argument("--num-workers", type=int)
    args = ap.parse_args()

    row = matrix_row(args.matrix, args.run_id)
    run_dir = args.runs_dir / row["run_id"]
    paths = {
        "fold_csv": fold_dir(args.folds_dir, row["organ"], row["variant"], row["design"])
        / f"fold_{row['fold']}.csv",
        "organ_config": args.configs / "cohort" / f"{row['organ']}.yaml",
        "model_config": args.configs / "training" / "models" / f"{row['aggregator']}.yaml",
        "trainer_config": args.configs / "training" / "trainer.yaml",
    }
    inputs = {name: sha256(path) for name, path in paths.items()}
    if clear_previous(run_dir, row, inputs):
        print(f"{row['run_id']} is complete; nothing to do")
        return

    organ_cfg = yaml.safe_load(paths["organ_config"].read_text())
    model_cfg = yaml.safe_load(paths["model_config"].read_text())
    trainer_cfg = yaml.safe_load(paths["trainer_config"].read_text())
    labels = list(organ_cfg["core_labels"])

    pl.seed_everything(int(row["seed"]), workers=True)
    data = SlideBagDataModule(
        paths["fold_csv"], labels, row["encoder"], batch_size=trainer_cfg["batch_size"],
        num_workers=args.num_workers if args.num_workers is not None
        else trainer_cfg["num_workers"],
        train_fraction=row["train_fraction"], train_fraction_seed=int(row["seed"]),
        stores_dir=args.stores_dir, max_dropped_share=trainer_cfg["max_dropped_share"])
    data.setup("fit")

    run_dir.mkdir(parents=True)
    run_record.write(run_dir, REPO, run_id=row["run_id"], families=row["families"].split(";"),
                     matrix_row=row, fold_csv=str(paths["fold_csv"]), inputs=inputs,
                     monitor=trainer_cfg["monitor"], device=device_name())
    try:
        write_json(run_dir / "config.json",
                   {"matrix_row": row, "labels": labels, "selection_labels": labels,
                    "fold_csv": str(paths["fold_csv"]), "input_dim": data.embed_dim,
                    "model": model_cfg, "trainer": trainer_cfg})
        write_json(run_dir / "label_mapping.json",
                   {"idx_to_label": {str(i): label for i, label in enumerate(labels)},
                    "selection_labels": labels})
        write_json(run_dir / "data_counts.json", data.counts())
        module = MILModule(
            row["aggregator"], data.embed_dim, labels, labels,
            architecture=model_cfg["architecture"], learning_rate=row["learning_rate"],
            weight_decay=model_cfg["weight_decay"], warmup_epochs=model_cfg["warmup_epochs"],
            instance_dropout=model_cfg["instance_dropout"],
            pos_weight_max=trainer_cfg["pos_weight_max"])
        trainer, checkpoint, early_stopping = build_trainer(run_dir, trainer_cfg)
        trainer.fit(module, datamodule=data)
        # Lightning may report the path through the resolved mount; the name locates the file.
        selected = run_dir / "checkpoints" / Path(checkpoint.best_model_path).name
        selected_epoch = torch.load(selected, map_location="cpu", weights_only=False)["epoch"]
        run_record.update(
            run_dir, status="complete", finished=run_record.now(),
            epochs_run=trainer.current_epoch, stopped_early=early_stopping.stopped_epoch > 0,
            selected_checkpoint=f"checkpoints/{selected.name}", selected_epoch=int(selected_epoch),
            val_ap_core_macro=float(checkpoint.best_model_score),
            pos_weight=module.pos_weight.tolist())
    except BaseException:
        run_record.update(run_dir, status="failed", finished=run_record.now())
        raise
    print(f"{row['run_id']}: selected {selected.name}")


if __name__ == "__main__":
    main()
