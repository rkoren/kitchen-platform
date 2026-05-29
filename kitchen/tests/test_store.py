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


# ---------------------------------------------------------------------------
# DataStore.preview
# ---------------------------------------------------------------------------


def test_preview_parquet_in_processed(tmp_path):
    store = DataStore(root=tmp_path)
    df = pd.DataFrame({"a": range(20)})
    store.save_parquet(df, "features.parquet")
    result = store.preview("features.parquet")
    assert len(result) == 5
    pd.testing.assert_frame_equal(result, df.head(5))


def test_preview_csv_in_raw(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    rows = "\n".join(["a,b"] + [f"{i},{i*2}" for i in range(10)])
    (store.raw_dir / "data.csv").write_text(rows)
    result = store.preview("data.csv")
    assert len(result) == 5


def test_preview_custom_n(tmp_path):
    store = DataStore(root=tmp_path)
    df = pd.DataFrame({"x": range(20)})
    store.save_parquet(df, "f.parquet")
    result = store.preview("f.parquet", n=3)
    assert len(result) == 3


def test_preview_processed_takes_priority_over_raw(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    (store.raw_dir / "data.csv").write_text("x\n99\n98\n")
    proc_df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]})
    store.save_parquet(proc_df, "data.parquet")
    # Two different filenames — put the same name in both stages
    store.raw_dir.mkdir(parents=True, exist_ok=True)
    (store.raw_dir / "data.parquet").write_bytes(
        (store.processed_dir / "data.parquet").read_bytes()
    )
    # Write a different frame to raw so we can tell them apart
    import io

    raw_df = pd.DataFrame({"x": [999]})
    buf = io.BytesIO()
    raw_df.to_parquet(buf)
    (store.raw_dir / "data.parquet").write_bytes(buf.getvalue())

    with pytest.warns(UserWarning, match="both processed/ and raw/"):
        result = store.preview("data.parquet")
    pd.testing.assert_frame_equal(result, proc_df.head(5))


def test_preview_warns_when_in_both_stages(tmp_path):
    store = DataStore(root=tmp_path)
    df = pd.DataFrame({"v": range(10)})
    store.save_parquet(df, "features.parquet")
    store.raw_dir.mkdir(parents=True, exist_ok=True)
    import io

    buf = io.BytesIO()
    df.to_parquet(buf)
    (store.raw_dir / "features.parquet").write_bytes(buf.getvalue())

    with pytest.warns(UserWarning, match="returning processed/ copy"):
        store.preview("features.parquet")


def test_preview_not_found_raises_with_listing(tmp_path):
    store = DataStore(root=tmp_path)
    store.save_parquet(pd.DataFrame({"a": [1]}), "other.parquet")
    store.raw_dir.mkdir(parents=True, exist_ok=True)
    (store.raw_dir / "train.csv").write_text("a\n1\n")

    with pytest.raises(FileNotFoundError) as exc_info:
        store.preview("missing.parquet")
    msg = str(exc_info.value)
    assert "missing.parquet" in msg
    assert "other.parquet" in msg
    assert "train.csv" in msg


def test_preview_not_found_no_data_dirs(tmp_path):
    store = DataStore(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="no data files found"):
        store.preview("anything.csv")


def test_preview_unsupported_extension(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    (store.raw_dir / "data.json").write_text("{}")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        store.preview("data.json")
