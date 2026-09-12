"""Loading checks and selection rules of the feature index."""

import pandas as pd
import pytest

from champs_pipeline.data_prep.feature_index import check_coverage, load_feature_index, usable_files


def make_index(rows):
    base = dict(encoder="virchow2", resolution="20x_224px_0px_overlap", wsi_path="K/S.svs", refused=None,
                level0_magnification=20.0, patch_size_level0=224.0, n_patches=10, wsi_objective_power=20.0)
    return pd.DataFrame([dict(base, **r) for r in rows])


def test_load_checks_columns_and_paths(tmp_path):
    path = tmp_path / "index.csv"
    make_index([dict(slide_id="S1", path="/f/S1.h5", timestamp="2026-01-01")]).drop(columns=["n_patches"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="n_patches"):
        load_feature_index(path)
    make_index([dict(slide_id="S1", path="/f/S1.h5", timestamp="2026-01-01"),
                dict(slide_id="S1", path="/f/S1.h5", timestamp="2026-01-02")]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="twice"):
        load_feature_index(path)


def test_usable_files_rules():
    index = make_index([
        dict(slide_id="S1", path="/a/S1.h5", timestamp="2026-01-01"),
        dict(slide_id="S1", path="/b/S1.h5", timestamp="2026-01-02"),
        dict(slide_id="S1", path="/c/S1.h5", timestamp="2026-01-03", refused="magnification differs from scanner"),
        dict(slide_id="S2", path="/a/S2.h5", timestamp="2026-01-01", refused="unreadable"),
        dict(slide_id="S3", path="/a/S3.h5", timestamp="2026-01-01", encoder="hoptimus0"),
    ])
    files = usable_files(index, "virchow2")
    assert list(files["path"]) == ["/b/S1.h5"]
    assert usable_files(index, "hoptimus0", "20x_256px_0px_overlap").empty
    check_coverage(index, ["S1"], "virchow2")
    with pytest.raises(ValueError, match="S2"):
        check_coverage(index, ["S1", "S2"], "virchow2")
