import pandas as pd
import pytest
from kitchen.store import DataStore


def test_standard_paths(tmp_path):
    store = DataStore(root=tmp_path)
    assert store.raw_dir == tmp_path / "data" / "raw"
    assert store.processed_dir == tmp_path / "data" / "processed"
    assert store.models_dir == tmp_path / "models"


def test_save_and_load_parquet(tmp_path):
    store = DataStore(root=tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    store.save_parquet(df, "features.parquet")
    result = store.load_parquet("features.parquet")
    pd.testing.assert_frame_equal(df, result)


def test_save_creates_directory(tmp_path):
    store = DataStore(root=tmp_path)
    df = pd.DataFrame({"x": [1]})
    store.save_parquet(df, "out.parquet")
    assert store.processed_dir.exists()


def test_load_csv(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    (store.raw_dir / "data.csv").write_text("a,b\n1,2\n3,4\n")
    df = store.load_csv("data.csv")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_parquet_missing_raises(tmp_path):
    store = DataStore(root=tmp_path)
    with pytest.raises(Exception):
        store.load_parquet("nonexistent.parquet")
