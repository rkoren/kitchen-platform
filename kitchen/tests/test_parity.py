"""CBB-027: champion ↔ processed-data feature parity."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from kitchen.parity import check_processed_parity, model_feature_names, parity_remedy


class _TwoHead:
    def __init__(self):
        self.margin_features = ["a", "b"]
        self.total_features = ["b", "c"]
        self._private_features = ["nope"]


def test_model_feature_names_unions_heads():
    assert model_feature_names(_TwoHead()) == ["a", "b", "c"]


def test_model_feature_names_sklearn_style_ndarray():
    m = type("M", (), {})()
    m.feature_names_in_ = np.array(["x", "y"], dtype=object)
    assert model_feature_names(m) == ["x", "y"]


def test_model_feature_names_ignores_mocks_and_non_lists():
    assert model_feature_names(MagicMock()) == []
    assert model_feature_names(object()) == []
    m = type("M", (), {})()
    m.features = "a,b"  # a string isn't a feature list
    m.n_features = 3
    assert model_feature_names(m) == []


def _parquet(dirpath, name, cols):
    pd.DataFrame({c: [1.0] for c in cols}).to_parquet(dirpath / name)


def test_parity_ok_when_one_file_has_everything(tmp_path):
    _parquet(tmp_path, "a.parquet", ["x"])
    _parquet(tmp_path, "b.parquet", ["x", "y", "z"])
    res = check_processed_parity(["x", "y"], tmp_path)
    assert res.ok and res.file.name == "b.parquet"


def test_parity_reports_closest_file(tmp_path):
    _parquet(tmp_path, "a.parquet", ["x"])
    _parquet(tmp_path, "b.parquet", ["x", "y"])
    res = check_processed_parity(["x", "y", "q"], tmp_path)
    assert not res.ok
    assert res.file.name == "b.parquet"
    assert res.missing == ("q",)


def test_parity_union_across_files_is_not_enough(tmp_path):
    _parquet(tmp_path, "a.parquet", ["x"])
    _parquet(tmp_path, "b.parquet", ["y"])
    assert not check_processed_parity(["x", "y"], tmp_path).ok


def test_parity_schema_file_pins_the_candidate(tmp_path):
    _parquet(tmp_path, "full.parquet", ["x", "y"])
    _parquet(tmp_path, "feat.parquet", ["x"])
    res = check_processed_parity(["x", "y"], tmp_path, schema_file="feat.parquet")
    assert not res.ok and res.file.name == "feat.parquet"


def test_parity_nothing_to_check(tmp_path):
    assert check_processed_parity([], tmp_path) is None
    assert check_processed_parity(["x"], tmp_path) is None  # no parquets
    assert check_processed_parity(["x"], tmp_path / "absent") is None


def test_parity_remedy_names_variant():
    assert "--variant rich" in parity_remedy("rich")
    assert "kitchen run features" in parity_remedy(None)
