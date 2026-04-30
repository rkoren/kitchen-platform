import os
from unittest.mock import MagicMock, patch

import mlflow
import pytest

from kitchen.tracking import Tracker, _flatten, configure, configure_from_env, init_experiment


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


def test_configure_from_env_defaults_to_local_file_store():
    with patch("kitchen.tracking.mlflow") as mock_mlflow:
        configure_from_env()
        mock_mlflow.set_tracking_uri.assert_called_once_with("./mlruns")


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
