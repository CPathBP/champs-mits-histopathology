"""Training on toy data: the data module, the aggregators, the matrix, the run lifecycle."""

import json
import os
import subprocess
import sys
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import yaml

from champs_pipeline.datasets.wsi.dataset import SlideBagDataModule
from champs_pipeline.datasets.wsi.learning_curve import case_order, subsample_train_by_fraction
from champs_pipeline.eval.run_record import git_state
from champs_pipeline.models.mil import AGGREGATORS
from champs_pipeline.models.mil.module import MILModule, masked_mean
from champs_pipeline.training.matrix import expand

REPO = Path(__file__).resolve().parents[1]
LABELS = ["finding_a", "finding_b"]
DIM = 8
# The toy store lacks features for one of four test slides.
TOY_DROP = 0.5


def write_store(path, slide_ids, rng):
    """A feature store with the columns of the cohort stage."""
    tiles = [int(rng.integers(3, 9)) for _ in slide_ids]
    table = pa.table({
        "slide_id": slide_ids,
        "features": pa.array([rng.normal(size=n * DIM).astype(np.float32).tolist() for n in tiles],
                             type=pa.large_list(pa.float32())),
        "embed_dim": pa.array([DIM] * len(slide_ids), type=pa.int32()),
        "num_patches": pa.array(tiles, type=pa.int32()),
    })
    lance.write_dataset(table, str(path))


@pytest.fixture()
def toy_fold(tmp_path):
    """Sixteen slides of eight cases, one without features, and the fold manifest."""
    rng = np.random.default_rng(0)
    slides = [f"S{i:02d}" for i in range(16)]
    store = tmp_path / "virchow2.lance"
    write_store(store, slides[:15], rng)
    split = ["train"] * 8 + ["val"] * 4 + ["test"] * 4
    frame = pd.DataFrame({
        "slide_id": slides, "case_id": [f"C{i // 2}" for i in range(16)], "split": split,
        "label_finding_a": [1.0, 0.0] * 8,
        "mask_finding_a": [1.0] * 16,
        "label_finding_b": [0.0, 1.0, 0.0, None] * 4,
        "mask_finding_b": [1.0, 1.0, 1.0, 0.0] * 4,
        "lance_dataset_path_virchow2": [str(store)] * 15 + [None],
        "lance_row_idx_virchow2": list(range(15)) + [None],
    })
    fold_dir = tmp_path / "folds" / "toy_elig_fivefold"
    fold_dir.mkdir(parents=True)
    frame.to_csv(fold_dir / "fold_0.csv", index=False)
    return tmp_path, fold_dir / "fold_0.csv"


def toy_data(fold_csv, **kwargs):
    options = {"num_workers": 0, "max_dropped_share": TOY_DROP, **kwargs}
    data = SlideBagDataModule(fold_csv, LABELS, "virchow2", **options)
    data.setup("fit")
    return data


def rewrite(fold_csv, change):
    frame = pd.read_csv(fold_csv)
    change(frame)
    frame.to_csv(fold_csv, index=False)


def test_data_module_serves_bags_masks_and_counts(toy_fold):
    _, fold_csv = toy_fold
    data = toy_data(fold_csv, batch_size=2)
    assert data.embed_dim == DIM
    counts = data.counts()
    assert counts["train"]["slides"] == 8 and counts["test"]["dropped_no_features"] == 1
    assert counts["train"]["labels"]["finding_b"] == {"pos": 2, "neg": 4, "masked": 2}
    item = data.datasets["val"][0]
    assert item["features"].shape[1] == DIM and item["label_mask"].tolist() == [1.0, 1.0]
    batch = next(iter(data.train_dataloader()))
    assert len(batch["features"]) == 2 and batch["label"].shape == (2, 2)


def test_loader_workers_read_the_store_the_main_process_opened(toy_fold):
    _, fold_csv = toy_fold
    data = toy_data(fold_csv, batch_size=3, num_workers=2)
    slides = [slide for batch in data.val_dataloader() for slide in batch["slide_id"]]
    assert slides == ["S08", "S09", "S10", "S11"]


def test_pos_weight_counts_kept_cells_only(toy_fold):
    _, fold_csv = toy_fold
    # finding_a: 4 positives, 4 negatives; finding_b: 2 positives, 4 kept negatives.
    assert toy_data(fold_csv).pos_weight().tolist() == [1.0, 2.0]


def test_a_store_row_of_another_slide_is_refused(toy_fold):
    _, fold_csv = toy_fold

    def point_first_slide_at_second_row(frame):
        frame.loc[0, "lance_row_idx_virchow2"] = 1

    rewrite(fold_csv, point_first_slide_at_second_row)
    with pytest.raises(ValueError, match="holds 'S01'"):
        toy_data(fold_csv).datasets["train"][0]


@pytest.mark.parametrize("label, mask, problem", [
    (1.0, 0.0, "a masked positive"),
    (None, 1.0, "an empty label that is kept"),
    (0.0, None, "an empty mask"),
])
def test_labels_that_contradict_their_masks_are_refused(toy_fold, label, mask, problem):
    _, fold_csv = toy_fold

    def change(frame):
        frame.loc[0, "label_finding_a"] = label
        frame.loc[0, "mask_finding_a"] = mask

    rewrite(fold_csv, change)
    with pytest.raises(ValueError, match=problem):
        toy_data(fold_csv)


def test_a_validation_split_without_a_class_is_refused(toy_fold):
    _, fold_csv = toy_fold

    def remove_validation_positives(frame):
        frame.loc[frame["split"] == "val", "label_finding_a"] = 0.0

    rewrite(fold_csv, remove_validation_positives)
    with pytest.raises(ValueError, match=r"val split lacks .*finding_a"):
        toy_data(fold_csv)


def test_an_encoder_missing_many_slides_is_refused(toy_fold):
    _, fold_csv = toy_fold
    with pytest.raises(ValueError, match="lacks features"):
        toy_data(fold_csv, max_dropped_share=0.05)


def test_learning_curve_fractions_are_nested_and_keep_whole_cases():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"case_id": [f"C{i // 2}" for i in range(200)],
                          "label_finding_a": rng.integers(0, 2, 200).astype(float),
                          "label_finding_b": (rng.random(200) < 0.1).astype(float)})
    small = subsample_train_by_fraction(frame, 0.2, LABELS, seed=3)
    large = subsample_train_by_fraction(frame, 0.55, LABELS, seed=3)
    assert set(small["case_id"]) < set(large["case_id"])
    assert small.groupby("case_id").size().eq(2).all()
    assert len(case_order(frame, LABELS, seed=3)) == 100
    assert subsample_train_by_fraction(frame, 1.0, LABELS) is frame


@pytest.mark.parametrize("name", ["meanpool", "abmil", "clam_mb", "acmil", "transmil"])
def test_aggregators_emit_one_logit_per_finding(name):
    config = yaml.safe_load((REPO / "configs/training/models" / f"{name}.yaml").read_text())
    architecture = dict(config["architecture"])
    if name == "acmil":
        architecture.update(n_masked_patch=3, diversity_weight=1.0)
    model = AGGREGATORS[name](input_dim=DIM, n_classes=3, **architecture).train()
    out = model(torch.randn(11, DIM))
    assert out["logits"].shape == (1, 3)
    if name == "acmil":
        assert out["extra_loss"].ndim == 0


def test_masked_cells_do_not_enter_the_loss():
    values = torch.tensor([[1.0, 100.0], [3.0, 5.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    assert masked_mean(values, mask).item() == pytest.approx(3.0)


def test_positive_weights_stay_out_of_the_checkpoint_state():
    module = MILModule("meanpool", DIM, LABELS, LABELS, architecture={"hidden_dim": 4})
    module.pos_weight = torch.tensor([2.0, 3.0])
    assert "pos_weight" not in module.state_dict()


def toy_matrix(**families):
    return {
        "defaults": {"organ": "lung", "variant": "elig", "design": "fivefold",
                     "encoder": "virchow2", "aggregator": "clam_mb", "train_fraction": 1.0,
                     "seed": 42},
        "families": families or {
            "headline": {"organ": ["lung", "liver"]},
            "aggregator_comparison": {"aggregator": ["abmil", "clam_mb"]},
            "learning_curve": {"organ": ["lung", "liver"], "train_fraction": [0.5, 1.0]},
        },
    }


MODELS = {"abmil": {"learning_rate": 1e-4}, "clam_mb": {"learning_rate": 5e-5}}


def test_matrix_trains_a_shared_run_once():
    rows = expand(toy_matrix(), MODELS, lambda *_: 5)
    assert rows["run_id"].is_unique
    assert len(rows) == 10 + 5 + 10  # headline, the new aggregator, the half fractions
    shared = rows[rows["families"] == "headline;aggregator_comparison;learning_curve"]
    assert len(shared) == 5 and set(shared["organ"]) == {"lung"}
    assert rows.loc[rows["aggregator"] == "abmil", "learning_rate"].eq(1e-4).all()
    assert len(expand(toy_matrix(), MODELS, lambda *_: 5, families=["headline"])) == 10


def test_matrix_refuses_two_runs_with_one_id():
    matrix = toy_matrix(learning_curve={"train_fraction": [0.13, 0.131]})
    with pytest.raises(ValueError, match="names two different runs"):
        expand(matrix, MODELS, lambda *_: 1)


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def test_untracked_code_makes_the_tree_dirty(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("x\n")
    git(tmp_path, "add", "README.md")
    git(tmp_path, "commit", "-q", "-m", "first")
    (tmp_path / "notes.txt").write_text("not code\n")
    assert git_state(tmp_path)[1] is False
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new_module.py").write_text("x = 1\n")
    assert git_state(tmp_path)[1] is True


def toy_configs(tmp_path):
    configs = tmp_path / "configs"
    (configs / "cohort").mkdir(parents=True)
    (configs / "training" / "models").mkdir(parents=True)
    (configs / "cohort" / "toy.yaml").write_text(yaml.safe_dump({"core_labels": LABELS}))
    model = yaml.safe_load((REPO / "configs/training/models/abmil.yaml").read_text())
    (configs / "training" / "models" / "abmil.yaml").write_text(yaml.safe_dump(model))
    trainer = yaml.safe_load((REPO / "configs/training/trainer.yaml").read_text())
    trainer.update(max_epochs=2, patience=2, precision="32-true", num_workers=0, batch_size=2,
                   max_dropped_share=TOY_DROP)
    (configs / "training" / "trainer.yaml").write_text(yaml.safe_dump(trainer))
    return configs


def train(tmp_path, root, configs):
    matrix = pd.DataFrame([{"run_id": "toy-run", "families": "headline", "organ": "toy",
                            "variant": "elig", "design": "fivefold", "fold": 0,
                            "encoder": "virchow2", "aggregator": "abmil",
                            "learning_rate": 1e-3, "train_fraction": 1.0, "seed": 42}])
    matrix.to_csv(tmp_path / "matrix.csv", index=False)
    command = [sys.executable, str(REPO / "scripts/training/train.py"),
               "--matrix", str(tmp_path / "matrix.csv"), "--run-id", "toy-run",
               "--folds-dir", str(root / "folds"), "--runs-dir", str(tmp_path / "runs"),
               "--configs", str(configs)]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("SLURM_")}
    return subprocess.run(command, check=True, capture_output=True, text=True,
                          env={**environment, "CUDA_VISIBLE_DEVICES": ""})


def test_a_run_trains_reloads_and_retries(toy_fold, tmp_path):
    root, _ = toy_fold
    configs = toy_configs(tmp_path)
    # The runs directory is reached through a symlink, as on a cluster mount.
    (tmp_path / "runs_storage").mkdir()
    (tmp_path / "runs").symlink_to(tmp_path / "runs_storage")
    train(tmp_path, root, configs)
    run_dir = tmp_path / "runs" / "toy-run"
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "complete" and set(record["inputs"]) == {
        "fold_csv", "organ_config", "model_config", "trainer_config"}
    selected = run_dir / record["selected_checkpoint"]
    MILModule.load_from_checkpoint(selected, map_location="cpu", strict=True)
    assert json.loads((run_dir / "label_mapping.json").read_text())["idx_to_label"]["1"] \
        == "finding_b"
    assert "val_ap_core_macro" in pd.read_csv(run_dir / "metrics.csv").columns

    assert "nothing to do" in train(tmp_path, root, configs).stdout

    record["status"] = "failed"
    (run_dir / "run.json").write_text(json.dumps(record))
    assert "moved to" in train(tmp_path, root, configs).stdout
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"
    assert len(list((tmp_path / "runs" / ".attempts" / "toy-run").iterdir())) == 1
