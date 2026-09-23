"""Selected-checkpoint routing, source guards and artifact completeness on toy data."""

import json

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import yaml

from champs_pipeline.eval import inference, load
from champs_pipeline.eval.predictions import check_cells, validate_predictions
from champs_pipeline.models.mil.module import MILModule


@pytest.fixture
def run_inputs(tmp_path):
    """A synthetic complete run with two ordered heads and one masked cell per split."""
    labels = ["finding_b", "finding_a"]
    slides = [f"slide-{i}" for i in range(6)]
    store = tmp_path / "features.lance"
    lance.write_dataset(pa.table({
        "slide_id": slides,
        "features": pa.array([np.arange(24, dtype=np.float32).tolist()] * 6,
                             type=pa.large_list(pa.float32())),
        "embed_dim": [8] * 6, "num_patches": [3] * 6,
    }), str(store))
    fold = tmp_path / "folds/lung_elig_fivefold/fold_0.csv"
    fold.parent.mkdir(parents=True)
    frame = pd.DataFrame({
        "slide_id": slides, "case_id": [f"case-{i // 2}" for i in range(6)],
        "split": ["train", "train", "val", "val", "test", "test"],
        "label_finding_b": [1., 0.] * 3, "mask_finding_b": [1] * 6,
        "label_finding_a": [0., None] * 3, "mask_finding_a": [1, 0] * 3,
        "lance_dataset_path_virchow2": [str(store)] * 6,
        "lance_row_idx_virchow2": list(range(6)),
    })
    frame.to_csv(fold, index=False)
    row = {"run_id": "toy-run", "families": "headline;learning_curve", "organ": "lung",
           "variant": "elig", "design": "fivefold", "fold": 0, "encoder": "virchow2",
           "aggregator": "meanpool", "learning_rate": 0.001, "train_fraction": 1., "seed": 42}
    directory = tmp_path / "runs/toy-run"
    (directory / "checkpoints").mkdir(parents=True)
    module = MILModule("meanpool", 8, labels, labels,
                       architecture={"hidden_dim": 4, "dropout": 0.5}, learning_rate=0.001)
    hp = dict(module.hparams)
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": hp, "epoch": 1},
               directory / "checkpoints/selected.ckpt")
    (directory / "hparams.yaml").write_text(yaml.safe_dump(hp))
    (directory / "label_mapping.json").write_text(json.dumps({
        "idx_to_label": {"0": labels[0], "1": labels[1]}, "selection_labels": labels}))
    record = {"run_id": row["run_id"], "status": "complete", "git_commit": "toy-commit",
              "git_dirty": False, "matrix_row": row, "selected_epoch": 1,
              "selected_checkpoint": "checkpoints/selected.ckpt",
              "inputs": {"fold_csv": inference.sha256(fold)}}
    (directory / "run.json").write_text(json.dumps(record))
    entry = {**row, **{k: record[k] for k in ("status", "git_commit", "git_dirty",
                                              "selected_epoch", "selected_checkpoint")}}
    matrix, registry = tmp_path / "matrix.csv", tmp_path / "registry.csv"
    pd.DataFrame([row]).to_csv(matrix, index=False)
    pd.DataFrame([entry]).to_csv(registry, index=False)
    return {"root": tmp_path, "row": entry, "directory": directory, "fold": fold,
            "registry": registry, "matrix": matrix, "module": module}


def prepare(inputs):
    return inference.prepare_run(inputs["row"], inputs["root"] / "runs", inputs["root"] / "folds")


def change_fold(inputs, change):
    frame = pd.read_csv(inputs["fold"])
    change(frame)
    frame.to_csv(inputs["fold"], index=False)
    path = inputs["directory"] / "run.json"
    record = json.loads(path.read_text())
    record["inputs"]["fold_csv"] = inference.sha256(inputs["fold"])
    path.write_text(json.dumps(record))


def test_round_trip_preserves_heads_masks_splits_and_full_bag_logits(run_inputs):
    context = prepare(run_inputs)
    predicted = inference.predict_run(context)
    assert set(predicted["split"]) == {"val", "test"}
    assert len(predicted) == 8
    assert predicted.loc[predicted["mask"].eq(0), "y_true"].isna().all()
    model = run_inputs["module"].model.eval()
    with torch.inference_mode():
        direct = model(torch.arange(24, dtype=torch.float32).reshape(3, 8))["logits"][0].numpy()
    for _, group in predicted.groupby("slide_id"):
        np.testing.assert_allclose(group.set_index("finding").loc[context["labels"], "logit"],
                                   direct, atol=1e-7)
    path = run_inputs["root"] / "roundtrip.parquet"
    predicted.to_parquet(path, index=False)
    checked = load.predictions(path).check(inference.expected_cells(context)).frame
    assert len(checked) == len(predicted)
    # Dropout is disabled, including when inference is called repeatedly.
    pd.testing.assert_frame_equal(predicted, inference.predict_run(context))


@pytest.mark.parametrize("defect", ["missing", "extra", "wrong_label", "wrong_case"])
def test_exact_coverage_and_source_equality(run_inputs, defect):
    context = prepare(run_inputs)
    frame = inference.predict_run(context)
    expected = inference.expected_cells(context)
    if defect == "missing":
        frame = frame.iloc[1:]
    elif defect == "extra":
        frame = pd.concat([frame, frame.iloc[:1].assign(slide_id="unexpected")])
    elif defect == "wrong_label":
        frame.loc[0, "y_true"] = 0
    else:
        frame.loc[0, "case_id"] = "another"
    with pytest.raises(ValueError):
        check_cells(frame, expected)


@pytest.mark.parametrize("column,value", [("score", np.inf), ("score", 0.12345),
                                          ("mask", 0), ("split", "train"), ("fold", 0.5)])
def test_invalid_cells_are_rejected(run_inputs, column, value):
    frame = inference.predict_run(prepare(run_inputs))
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_predictions(frame)


def test_missing_matrix_run_is_rejected(run_inputs):
    matrix = pd.read_csv(run_inputs["matrix"])
    pd.concat([matrix, matrix.assign(run_id="missing")]).to_csv(run_inputs["matrix"], index=False)
    with pytest.raises(ValueError, match="complete training matrix"):
        inference.prepare_registry(run_inputs["registry"], run_inputs["root"] / "runs",
                                   run_inputs["root"] / "folds", run_inputs["matrix"])


def test_dirty_training_and_registry_identity_mismatch_are_rejected(run_inputs):
    path = run_inputs["directory"] / "run.json"
    record = json.loads(path.read_text())
    record["git_dirty"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="clean commit"):
        prepare(run_inputs)
    record["git_dirty"] = False
    path.write_text(json.dumps(record))
    run_inputs["row"]["selected_epoch"] = 2
    with pytest.raises(ValueError, match="registry selected_epoch"):
        prepare(run_inputs)


def test_changed_fold_and_head_permutation_are_rejected(run_inputs):
    original = run_inputs["fold"].read_bytes()
    run_inputs["fold"].write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="fold hash"):
        prepare(run_inputs)
    run_inputs["fold"].write_bytes(original)
    path = run_inputs["directory"] / "label_mapping.json"
    mapping = json.loads(path.read_text())
    mapping["idx_to_label"] = {"0": "finding_a", "1": "finding_b"}
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="checkpoint labels"):
        prepare(run_inputs)


@pytest.mark.parametrize("defect", ["missing_features", "wrong_row", "case_leak"])
def test_fold_guards(run_inputs, defect):
    def change(frame):
        if defect == "missing_features":
            frame.loc[2, "lance_dataset_path_virchow2"] = None
        elif defect == "wrong_row":
            frame.loc[2, "lance_row_idx_virchow2"] = 0
        else:
            frame.loc[2, "case_id"] = frame.loc[0, "case_id"]
    change_fold(run_inputs, change)
    with pytest.raises(ValueError):
        prepare(run_inputs)


def test_strict_weights_and_changed_preflight_inputs(run_inputs):
    context = prepare(run_inputs)
    path = run_inputs["directory"] / "checkpoints/selected.ckpt"
    state = torch.load(path, weights_only=False)
    state["state_dict"]["unexpected_weight"] = torch.zeros(1)
    torch.save(state, path)
    with pytest.raises(ValueError, match="input changed"):
        inference.predict_run(context)
    context = prepare(run_inputs)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        inference.predict_run(context)


def test_atomic_publication_record_and_interrupted_rerun(run_inputs, monkeypatch):
    contexts = inference.prepare_registry(run_inputs["registry"], run_inputs["root"] / "runs",
                                         run_inputs["root"] / "folds", run_inputs["matrix"])
    # Simulated clean git identity, isolated to this temporary fixture.
    monkeypatch.setattr(inference.run_record, "git_state", lambda _: ("toy-inference", False))
    path = run_inputs["root"] / "predictions.parquet"
    inference.write_predictions(contexts, path, run_inputs["root"],
                                run_inputs["registry"], run_inputs["matrix"])
    assert len(load.predictions(path).check().frame) == 8
    original = path.read_bytes()
    def fail(*args, **kwargs):
        raise RuntimeError("interrupted inference")
    monkeypatch.setattr(inference, "predict_run", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        inference.write_predictions(contexts, path, run_inputs["root"],
                                    run_inputs["registry"], run_inputs["matrix"])
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".predictions.parquet.*"))
    changed = pd.read_parquet(path).iloc[:-1]
    changed.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="matching complete"):
        load.predictions(path).check()



def test_liver_and_raw_cells_use_the_same_contract(run_inputs):
    frame = inference.predict_run(prepare(run_inputs))
    frame = frame.assign(organ_group="liver", label_variant="raw", design="loso_nested")
    assert len(validate_predictions(frame)) == 8


def test_wrong_embedding_width_is_rejected(run_inputs):
    store = run_inputs["root"] / "narrow.lance"
    slides = pd.read_csv(run_inputs["fold"])["slide_id"].tolist()
    lance.write_dataset(pa.table({
        "slide_id": slides,
        "features": pa.array([np.arange(18, dtype=np.float32).tolist()] * len(slides),
                             type=pa.large_list(pa.float32())),
        "embed_dim": [6] * len(slides), "num_patches": [3] * len(slides),
    }), str(store))
    change_fold(run_inputs, lambda frame: frame.__setitem__("lance_dataset_path_virchow2",
                                                            str(store)))
    with pytest.raises(ValueError, match="embedding width"):
        inference.predict_run(prepare(run_inputs))


def test_kept_cell_with_null_label_is_rejected(run_inputs):
    frame = inference.predict_run(prepare(run_inputs))
    kept = frame.index[frame["mask"].eq(1)][0]
    frame.loc[kept, "y_true"] = np.nan
    with pytest.raises(ValueError, match="kept labels"):
        validate_predictions(frame)
    change_fold(run_inputs, lambda frame: frame.__setitem__(
        "label_finding_b", [np.nan] + frame["label_finding_b"].tolist()[1:]))
    with pytest.raises(ValueError, match="empty label that is kept"):
        prepare(run_inputs)


def test_duplicate_registry_run_and_fold_slide_are_rejected(run_inputs):
    registry = pd.read_csv(run_inputs["registry"])
    pd.concat([registry, registry]).to_csv(run_inputs["registry"], index=False)
    with pytest.raises(ValueError, match="duplicate run ids"):
        inference.prepare_registry(run_inputs["registry"], run_inputs["root"] / "runs",
                                   run_inputs["root"] / "folds", run_inputs["matrix"])
    original = pd.read_csv(run_inputs["fold"])
    change_fold(run_inputs, lambda frame: frame.loc.__setitem__(len(frame), original.iloc[2]))
    with pytest.raises(ValueError, match="duplicate slides"):
        prepare(run_inputs)


def test_draft_check_skips_only_the_inference_record(run_inputs):
    path = run_inputs["root"] / "draft.parquet"
    frame = inference.predict_run(prepare(run_inputs))
    frame.to_parquet(path, index=False)
    with pytest.raises(FileNotFoundError):
        load.predictions(path).check()
    assert len(load.predictions(path).check(draft=True).frame) == 8
    frame.assign(score=0.12345).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="sigmoid"):
        load.predictions(path).check(draft=True)


def test_checker_binds_artifact_to_current_checkpoint(run_inputs, monkeypatch):
    contexts = [prepare(run_inputs)]
    monkeypatch.setattr(inference.run_record, "git_state", lambda _: ("toy-inference", False))
    path = run_inputs["root"] / "predictions.parquet"
    inference.write_predictions(contexts, path, run_inputs["root"],
                                run_inputs["registry"], run_inputs["matrix"])
    inference.check_provenance(path, contexts, run_inputs["registry"], run_inputs["matrix"])
    checkpoint = contexts[0]["checkpoint"]
    state = torch.load(checkpoint, weights_only=False)
    first_key = next(iter(state["state_dict"]))
    state["state_dict"][first_key] = state["state_dict"][first_key] + 0.01
    torch.save(state, checkpoint)
    current = [prepare(run_inputs)]
    # Cell identities and the artifact's own receipt still pass; source binding must not.
    inference.check_artifact(path, current)
    load.predictions(path).check()
    with pytest.raises(ValueError, match="supplied run inputs"):
        inference.check_provenance(path, current, run_inputs["registry"], run_inputs["matrix"])
    with pytest.raises(ValueError, match="registry or matrix"):
        inference.check_provenance(path, contexts, run_inputs["matrix"], run_inputs["registry"])
