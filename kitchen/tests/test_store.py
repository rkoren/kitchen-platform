import pandas as pd
import pytest
from kitchen.store import DataStore


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

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


def test_save_parquet_models_stage(tmp_path):
    store = DataStore(root=tmp_path)
    df = pd.DataFrame({"pred": [0.9, 0.1]})
    path = store.save_parquet(df, "preds.parquet", stage="models")
    assert path == store.models_dir / "preds.parquet"
    assert path.exists()


# ---------------------------------------------------------------------------
# root validation
# ---------------------------------------------------------------------------

def test_root_nonexistent_raises(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        DataStore(root=missing)


def test_root_none_uses_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    store = DataStore()
    assert store.root == tmp_path


# ---------------------------------------------------------------------------
# Invalid stage
# ---------------------------------------------------------------------------

def test_save_parquet_invalid_stage_raises(tmp_path):
    store = DataStore(root=tmp_path)
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="Unknown stage"):
        store.save_parquet(df, "out.parquet", stage="typo")


def test_load_parquet_invalid_stage_raises(tmp_path):
    store = DataStore(root=tmp_path)
    with pytest.raises(ValueError, match="Unknown stage"):
        store.load_parquet("out.parquet", stage="bad_stage")


# ---------------------------------------------------------------------------
# Actionable errors on missing files
# ---------------------------------------------------------------------------

def test_load_parquet_missing_raises_with_command(tmp_path):
    store = DataStore(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="kitchen run features"):
        store.load_parquet("nonexistent.parquet")


def test_load_csv_missing_raises_with_command(tmp_path):
    store = DataStore(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="kitchen ingest"):
        store.load_csv("missing.csv")


def test_load_parquet_raw_stage_missing_raises_with_command(tmp_path):
    store = DataStore(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="kitchen ingest"):
        store.load_parquet("data.parquet", stage="raw")


def test_load_parquet_models_stage_missing_raises_with_command(tmp_path):
    store = DataStore(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="kitchen run train"):
        store.load_parquet("model.parquet", stage="models")
