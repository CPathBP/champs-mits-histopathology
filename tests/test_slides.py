"""The slide scripts on synthetic inputs."""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import crawl_wsi
import train_stain_classifier

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "slides"


def run(script, *args):
    subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], check=True)


def write_h5(path, n, level0, tile0, dim=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("features", data=np.zeros((n, dim), dtype=np.float32))
        coords = h.create_dataset("coords", data=np.zeros((n, 2), dtype=np.int64))
        coords.attrs["level0_magnification"] = level0
        coords.attrs["patch_size_level0"] = tile0


@pytest.mark.parametrize("slide_id, label", [
    ("2021-1218-B_HE_SLCC00013_M02965.044", "he"),
    ("M02905.043 - 2021-08-13 16.44.47", "he"),
    ("2019-0260-B_GRAM_BDAA00751_M00982.044", "gram"),
    ("2019-0260-B_IHC2017-069_BDAA00751", "ihc"),
    ("2019-0260-B_RETICULIN_BDAA00751", "other"),
])
def test_stain_label(slide_id, label):
    assert train_stain_classifier.stain_label(slide_id) == label


def test_crawl(tmp_path):
    case = tmp_path / "Kenya" / "Cases" / "K1"
    case.mkdir(parents=True)
    (case / "a.svs").write_bytes(b"")
    (case / "2019-0260-B_HE_KEAA00002_M00982.044.svs").write_bytes(b"")
    (case / "notes.txt").write_bytes(b"")
    run("crawl_wsi.py", "--data-root", tmp_path, "--out", tmp_path / "inv.csv", "--workers", 1)
    inv = pd.read_csv(tmp_path / "inv.csv")
    assert list(inv["slide_id"]) == ["2019-0260-B_HE_KEAA00002_M00982.044", "a"]
    assert (inv["site"] == "Kenya").all() and (inv["case_id"] == "K1").all()
    assert list(inv["id_conflict"]) == [True, False]
    assert inv["probe_error"].notna().all()
    assert crawl_wsi.probe(str(case / "a.svs"))["probe_error"]


def test_index_and_chunks(tmp_path):
    root = tmp_path / "features"
    res = "20x_224px_0px_overlap"
    write_h5(root / "virchow2" / "kenya" / "chunk_0" / res / "features_virchow2" / "S1.h5", 10, 40, 448)
    write_h5(root / "virchow2" / "kenya" / "chunk_1" / res / "features_virchow2" / "S1.h5", 10, 40, 448)
    write_h5(root / "hoptimus0" / "kenya" / "chunk_0" / res / "features_hoptimus0" / "S1.h5", 40, 20, 224)
    inventory = tmp_path / "inv.csv"
    pd.DataFrame({"slide_id": ["S1", "S2"], "wsi_path": ["Kenya/Cases/K1/S1.svs", "Kenya/Cases/K1/S2.svs"],
                  "site": "Kenya", "case_id": "K1", "id_conflict": False, "scanner_power": 40.0,
                  "probe_error": None}).to_csv(inventory, index=False)
    run("index_features.py", "--features-root", root, "--inventory", inventory, "--out", tmp_path / "index.csv", "--workers", 2)
    index = pd.read_csv(tmp_path / "index.csv")
    assert len(index) == 3 and index["wsi_objective_power"].eq(40.0).all()
    assert index.loc[index.encoder == "hoptimus0", "refused"].iloc[0] == "magnification differs from scanner"
    assert index.loc[index.encoder == "virchow2", "refused"].isna().all()
    assert set(index.columns) >= {"slide_id", "encoder", "resolution", "path", "timestamp", "level0_magnification",
                                  "patch_size_level0", "n_patches", "wsi_path", "site", "case_id"}

    run("chunk_manifests.py", "--inventory", inventory, "--site", "Kenya", "--out-dir", tmp_path / "chunks", "--chunk-size", 1)
    chunks = sorted((tmp_path / "chunks").glob("*.csv"))
    assert [c.name for c in chunks] == ["kenya_chunk_0.csv", "kenya_chunk_1.csv"]
    assert list(pd.read_csv(chunks[0]).columns) == ["wsi", "patient_id", "notes"]

    run("chunk_manifests.py", "--inventory", inventory, "--index", tmp_path / "index.csv", "--coords-encoder", "virchow2",
        "--name", "indomain", "--out-dir", tmp_path / "reuse")
    reuse = pd.read_csv(tmp_path / "reuse" / "indomain_chunk_0.csv")
    assert list(reuse["wsi"]) == ["Kenya/Cases/K1/S1.svs"]
    assert reuse["coords_path"].iloc[0].endswith("chunk_1/20x_224px_0px_overlap/patches/S1_patches.h5")
