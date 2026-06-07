from unittest.mock import patch

import pandas as pd
import pytest

from kitchen.store import DataStore, SchemaError

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


# ---------------------------------------------------------------------------
# DataStore.list
# ---------------------------------------------------------------------------


def test_list_raw_returns_sorted_filenames(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    for name in ["c.csv", "a.csv", "b.csv"]:
        (store.raw_dir / name).write_text("x\n1\n")
    assert store.list("raw") == ["a.csv", "b.csv", "c.csv"]


def test_list_processed_returns_filenames(tmp_path):
    store = DataStore(root=tmp_path)
    df = pd.DataFrame({"x": [1, 2]})
    store.save_parquet(df, "features.parquet")
    assert store.list("processed") == ["features.parquet"]


def test_list_missing_dir_returns_empty_list(tmp_path):
    store = DataStore(root=tmp_path)
    assert store.list("raw") == []


def test_list_empty_dir_returns_empty_list(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    assert store.list("raw") == []


def test_list_excludes_subdirectories(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    (store.raw_dir / "train.csv").write_text("x\n1\n")
    (store.raw_dir / "subdir").mkdir()
    assert store.list("raw") == ["train.csv"]


def test_list_custom_relative_path(tmp_path):
    store = DataStore(root=tmp_path)
    custom = tmp_path / "data" / "external"
    custom.mkdir(parents=True)
    (custom / "extra.csv").write_text("x\n1\n")
    assert store.list("data/external") == ["extra.csv"]


def test_list_custom_path_missing_returns_empty(tmp_path):
    store = DataStore(root=tmp_path)
    assert store.list("data/external") == []


def test_list_default_stage_is_raw(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    (store.raw_dir / "train.csv").write_text("x\n1\n")
    assert store.list() == ["train.csv"]


# ── DataStore.is_stale ────────────────────────────────────────────────────────


def test_is_stale_returns_true_when_output_missing(tmp_path):
    store = DataStore(root=tmp_path)
    dep = tmp_path / "data" / "raw" / "train.csv"
    dep.parent.mkdir(parents=True)
    dep.write_text("x\n1\n")
    assert store.is_stale("data/processed/features.parquet", ["data/raw/train.csv"]) is True


def test_is_stale_returns_false_when_output_is_newer(tmp_path):
    import time

    store = DataStore(root=tmp_path)
    dep = tmp_path / "data" / "raw" / "train.csv"
    dep.parent.mkdir(parents=True)
    dep.write_text("x\n1\n")

    time.sleep(0.01)
    out = tmp_path / "data" / "processed" / "features.parquet"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"parquet")

    assert store.is_stale("data/processed/features.parquet", ["data/raw/train.csv"]) is False


def test_is_stale_returns_true_when_dep_is_newer(tmp_path):
    import time

    store = DataStore(root=tmp_path)
    out = tmp_path / "data" / "processed" / "features.parquet"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"parquet")

    time.sleep(0.01)
    dep = tmp_path / "data" / "raw" / "train.csv"
    dep.parent.mkdir(parents=True)
    dep.write_text("x\n1\n")

    assert store.is_stale("data/processed/features.parquet", ["data/raw/train.csv"]) is True


def test_is_stale_any_dep_triggers_stale(tmp_path):
    import time

    store = DataStore(root=tmp_path)
    out = tmp_path / "data" / "processed" / "features.parquet"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"old")

    dep_old = tmp_path / "data" / "raw" / "a.csv"
    dep_old.parent.mkdir(parents=True)
    dep_old.write_text("a")

    time.sleep(0.01)
    out.write_bytes(b"newer")  # output refreshed after dep_old

    time.sleep(0.01)
    dep_new = tmp_path / "data" / "raw" / "b.csv"  # this dep is newest of all
    dep_new.write_text("b")

    assert store.is_stale("data/processed/features.parquet", ["data/raw/a.csv", "data/raw/b.csv"]) is True


def test_is_stale_missing_dep_is_ignored(tmp_path):
    import time

    store = DataStore(root=tmp_path)
    out = tmp_path / "data" / "processed" / "features.parquet"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"parquet")

    time.sleep(0.01)
    # dep does not exist — should be skipped, not raise
    assert store.is_stale("data/processed/features.parquet", ["data/raw/nonexistent.csv"]) is False


def test_is_stale_accepts_absolute_paths(tmp_path):
    store = DataStore(root=tmp_path)
    dep = tmp_path / "train.csv"
    dep.write_text("x")
    # output missing — absolute dep path should still work
    assert store.is_stale(tmp_path / "features.parquet", [dep]) is True


def test_is_stale_empty_deps_returns_false_when_output_exists(tmp_path):
    store = DataStore(root=tmp_path)
    out = tmp_path / "data" / "processed" / "features.parquet"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"parquet")
    assert store.is_stale("data/processed/features.parquet", []) is False


# ---------------------------------------------------------------------------
# Schema validation (DS-002)
# ---------------------------------------------------------------------------


def _df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


def test_load_parquet_schema_match_passes(tmp_path):
    store = DataStore(root=tmp_path)
    store.save_parquet(_df(), "f.parquet")
    result = store.load_parquet("f.parquet", schema={"a": "int64", "b": "float64"})
    assert list(result.columns) == ["a", "b"]


def test_load_parquet_schema_dtype_mismatch_raises(tmp_path):
    store = DataStore(root=tmp_path)
    store.save_parquet(_df(), "f.parquet")
    with pytest.raises(SchemaError) as exc:
        store.load_parquet("f.parquet", schema={"a": "float64"})
    msg = str(exc.value)
    assert "a:" in msg and "float64" in msg and "int64" in msg


def test_load_parquet_schema_missing_column_raises(tmp_path):
    store = DataStore(root=tmp_path)
    store.save_parquet(_df(), "f.parquet")
    with pytest.raises(SchemaError, match="missing"):
        store.load_parquet("f.parquet", schema={"c": "int64"})


def test_load_parquet_schema_accepts_python_dtype(tmp_path):
    """Schema values may be plain Python types (int/float), not just strings."""
    store = DataStore(root=tmp_path)
    store.save_parquet(_df(), "f.parquet")
    result = store.load_parquet("f.parquet", schema={"a": int, "b": float})
    assert len(result) == 3


def test_load_csv_schema_mismatch_raises(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    _df().to_csv(store.raw_dir / "r.csv", index=False)
    with pytest.raises(SchemaError):
        store.load_csv("r.csv", schema={"a": "float64"})


def test_load_csv_schema_match_passes(tmp_path):
    store = DataStore(root=tmp_path)
    store.raw_dir.mkdir(parents=True)
    _df().to_csv(store.raw_dir / "r.csv", index=False)
    result = store.load_csv("r.csv", schema={"a": "int64", "b": "float64"})
    assert len(result) == 3


# ---------------------------------------------------------------------------
# load_parquet(run_id=...) — fetch from MLflow artifacts (DS-003)
# ---------------------------------------------------------------------------


def test_load_parquet_run_id_fetches_from_mlflow(tmp_path):
    """With run_id, the file is downloaded from MLflow artifacts, not the local stage."""
    # Stage a parquet that download_artifacts will "return".
    artifact = tmp_path / "downloaded.parquet"
    _df().to_parquet(artifact, index=False)

    store = DataStore(root=tmp_path)  # note: no local data/processed/f.parquet exists
    with patch(
        "mlflow.artifacts.download_artifacts", return_value=str(artifact)
    ) as mock_dl:
        result = store.load_parquet("f.parquet", run_id="abc123def456")

    mock_dl.assert_called_once()
    _, kwargs = mock_dl.call_args
    assert kwargs["run_id"] == "abc123def456"
    assert kwargs["artifact_path"] == "f.parquet"
    pd.testing.assert_frame_equal(result, _df())


def test_load_parquet_run_id_applies_schema(tmp_path):
    artifact = tmp_path / "downloaded.parquet"
    _df().to_parquet(artifact, index=False)
    store = DataStore(root=tmp_path)
    with patch("mlflow.artifacts.download_artifacts", return_value=str(artifact)):
        with pytest.raises(SchemaError):
            store.load_parquet("f.parquet", run_id="abc123def456", schema={"a": "float64"})
