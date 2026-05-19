import os
from unittest.mock import MagicMock, patch

import mlflow
import pytest

from kitchen.tracking import (
    Tracker,
    _dict_hash,
    _file_hash,
    _flatten,
    _git_sha,
    _package_versions,
    configure,
    configure_from_env,
    init_experiment,
    log_run_context,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_BUCKET", raising=False)


# ── Tracker ───────────────────────────────────────────────────────────────────

def test_flatten_nested():
    assert _flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


def test_flatten_already_flat():
    assert _flatten({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_tracker_logs_run(tmp_path):
    uri = f"file://{tmp_path}"
    tracker = Tracker("test-exp", tracking_uri=uri)
    with tracker.run(run_name="r1", params={"lr": 0.1}):
        pass

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 1
    assert runs.iloc[0]["params.lr"] == "0.1"


def test_tracker_log_metrics(tmp_path):
    uri = f"file://{tmp_path}"
    tracker = Tracker("test-exp", tracking_uri=uri)
    with tracker.run():
        tracker.log_metrics({"brier": 0.12, "auc": 0.88})

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert float(runs.iloc[0]["metrics.brier"]) == pytest.approx(0.12)


def test_tracker_log_model_unknown_flavour(tmp_path):
    uri = f"file://{tmp_path}"
    tracker = Tracker("test-exp", tracking_uri=uri)
    with tracker.run():
        with pytest.raises(ValueError, match="Unknown flavour"):
            tracker.log_model(object(), "model", flavour="tensorflow")


# ── configure ─────────────────────────────────────────────────────────────────

def test_configure_sets_tracking_uri():
    with patch("kitchen.tracking.mlflow") as mock_mlflow:
        configure("http://mlflow:5000")
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")


def test_configure_sets_artifact_bucket_env(monkeypatch):
    with patch("kitchen.tracking.mlflow"):
        configure("./mlruns", artifact_bucket="my-bucket")
        assert os.environ["MLFLOW_ARTIFACT_BUCKET"] == "my-bucket"


def test_configure_no_artifact_bucket_leaves_env_unset():
    with patch("kitchen.tracking.mlflow"):
        configure("./mlruns")
        assert "MLFLOW_ARTIFACT_BUCKET" not in os.environ


# ── configure_from_env ────────────────────────────────────────────────────────

def test_configure_from_env_reads_tracking_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://remote:5000")
    with patch("kitchen.tracking.mlflow") as mock_mlflow:
        configure_from_env()
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://remote:5000")


def test_configure_from_env_defaults_to_sqlite():
    with patch("kitchen.tracking.mlflow") as mock_mlflow:
        configure_from_env()
        mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///mlruns.db")


def test_configure_from_env_passes_artifact_bucket(monkeypatch):
    monkeypatch.setenv("MLFLOW_ARTIFACT_BUCKET", "my-bucket")
    with patch("kitchen.tracking.mlflow"):
        configure_from_env()
        assert os.environ["MLFLOW_ARTIFACT_BUCKET"] == "my-bucket"


# ── init_experiment ───────────────────────────────────────────────────────────

def test_init_experiment_returns_existing_id():
    mock_exp = MagicMock()
    mock_exp.experiment_id = "42"
    with patch("kitchen.tracking.mlflow") as mock_mlflow:
        mock_mlflow.get_experiment_by_name.return_value = mock_exp
        result = init_experiment("my-experiment")
        assert result == "42"
        mock_mlflow.create_experiment.assert_not_called()


def test_init_experiment_creates_when_missing():
    with patch("kitchen.tracking.mlflow") as mock_mlflow:
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "99"
        result = init_experiment("new-experiment")
        assert result == "99"
        mock_mlflow.create_experiment.assert_called_once()


def test_init_experiment_uses_s3_artifact_location(monkeypatch):
    monkeypatch.setenv("MLFLOW_ARTIFACT_BUCKET", "my-bucket")
    with patch("kitchen.tracking.mlflow") as mock_mlflow:
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "1"
        init_experiment("my-exp")
        _, kwargs = mock_mlflow.create_experiment.call_args
        assert kwargs["artifact_location"] == "s3://my-bucket/mlflow-artifacts/my-exp"


def test_init_experiment_no_artifact_location_without_bucket():
    with patch("kitchen.tracking.mlflow") as mock_mlflow:
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "1"
        init_experiment("my-exp")
        _, kwargs = mock_mlflow.create_experiment.call_args
        assert kwargs["artifact_location"] is None


# ── K-008: reproducibility helpers ───────────────────────────────────────────

def test_git_sha_returns_string_or_none():
    result = _git_sha()
    assert result is None or (isinstance(result, str) and len(result) == 40)


def test_git_sha_returns_none_on_failure():
    with patch("subprocess.run", side_effect=Exception("no git")):
        assert _git_sha() is None


def test_git_sha_returns_none_on_nonzero_exit():
    mock_result = MagicMock()
    mock_result.returncode = 128
    with patch("subprocess.run", return_value=mock_result):
        assert _git_sha() is None


def test_package_versions_returns_known_package():
    versions = _package_versions(["mlflow"])
    assert "mlflow" in versions
    assert isinstance(versions["mlflow"], str)


def test_package_versions_skips_missing_package():
    versions = _package_versions(["mlflow", "no-such-pkg-xyz"])
    assert "mlflow" in versions
    assert "no-such-pkg-xyz" not in versions


def test_dict_hash_is_deterministic():
    d = {"b": 2, "a": 1}
    assert _dict_hash(d) == _dict_hash(d)


def test_dict_hash_order_independent():
    assert _dict_hash({"a": 1, "b": 2}) == _dict_hash({"b": 2, "a": 1})


def test_dict_hash_differs_on_different_content():
    assert _dict_hash({"a": 1}) != _dict_hash({"a": 2})


def test_file_hash_matches_sha256(tmp_path):
    import hashlib
    content = b"hello world"
    f = tmp_path / "data.bin"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert _file_hash(f) == expected


def test_log_run_context_sets_tags(tmp_path):
    uri = f"file://{tmp_path}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("ctx-test")
    with mlflow.start_run():
        log_run_context(params={"lr": 0.1})

    runs = mlflow.search_runs(experiment_names=["ctx-test"])
    tags = runs.iloc[0]
    assert tags["tags.kitchen.python"].count(".") == 2
    assert len(tags["tags.kitchen.params_sha256"]) == 64


def test_log_run_context_logs_data_hash(tmp_path):
    data_file = tmp_path / "features.parquet"
    data_file.write_bytes(b"fake parquet content")
    uri = f"file://{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("ctx-data-test")
    with mlflow.start_run():
        log_run_context(data_path=data_file)

    runs = mlflow.search_runs(experiment_names=["ctx-data-test"])
    assert len(runs.iloc[0]["tags.kitchen.data_sha256"]) == 64


def test_log_run_context_skips_missing_data_file(tmp_path):
    uri = f"file://{tmp_path}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("ctx-missing-test")
    with mlflow.start_run():
        log_run_context(data_path=tmp_path / "does_not_exist.parquet")

    runs = mlflow.search_runs(experiment_names=["ctx-missing-test"])
    assert "tags.kitchen.data_sha256" not in runs.columns or runs.iloc[0].get("tags.kitchen.data_sha256") != runs.iloc[0].get("tags.kitchen.data_sha256")  # NaN


def test_log_run_context_is_best_effort(tmp_path):
    uri = f"file://{tmp_path}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("ctx-err-test")
    with mlflow.start_run():
        with patch("kitchen.tracking.mlflow.set_tags", side_effect=Exception("boom")):
            log_run_context(params={"x": 1})  # must not raise


def test_tracker_run_sets_context_tags(tmp_path):
    uri = f"file://{tmp_path}"
    tracker = Tracker("ctx-tracker-test", tracking_uri=uri)
    with tracker.run(params={"depth": 3}):
        pass

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["ctx-tracker-test"])
    assert "tags.kitchen.python" in runs.columns
