import os
from unittest.mock import MagicMock, patch

import mlflow
import pytest

from kitchen.tracking import (
    ArtifactLocationError,
    MlflowSchemaError,
    Tracker,
    _dict_hash,
    _file_hash,
    _flatten,
    _git_sha,
    _is_schema_outdated,
    _local_path_from_uri,
    _package_versions,
    _schema_remediation,
    _verify_store_schema,
    configure,
    configure_from_env,
    explain_model_load_error,
    init_experiment,
    log_run_context,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_BUCKET", raising=False)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")


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


def test_configure_sets_artifact_bucket_env(monkeypatch):  # pylint: disable=unused-argument
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


# ── CBB-001: out-of-date schema translation ────────────────────────────────────


def test_is_schema_outdated_matches_known_messages():
    assert _is_schema_outdated(Exception("Detected out-of-date database schema (found ...)"))
    assert _is_schema_outdated(Exception("please run 'mlflow db upgrade <uri>'"))
    assert not _is_schema_outdated(Exception("connection refused"))


def test_schema_remediation_sqlite_includes_archive_hint():
    msg = _schema_remediation("sqlite:///mlruns.db")
    assert "mlflow db upgrade sqlite:///mlruns.db" in msg
    assert "mv mlruns.db mlruns.db.bak" in msg


def test_schema_remediation_remote_omits_archive_hint():
    msg = _schema_remediation("postgresql://host/db")
    assert "mlflow db upgrade" in msg
    assert "\n         mv " not in msg  # no sqlite archive command


def test_verify_store_schema_translates_outdated(tmp_path):
    db = tmp_path / "mlruns.db"
    db.write_text("x")  # exists → probe runs
    exc = mlflow.exceptions.MlflowException(
        "Detected out-of-date database schema (found version a, but expected b). "
        "run 'mlflow db upgrade'"
    )
    with patch("kitchen.tracking.mlflow.search_experiments", side_effect=exc):
        with pytest.raises(MlflowSchemaError) as ei:
            _verify_store_schema(f"sqlite:///{db}")
    assert "mlflow db upgrade" in str(ei.value)


def test_verify_store_schema_ignores_non_schema_errors(tmp_path):
    db = tmp_path / "mlruns.db"
    db.write_text("x")
    with patch(
        "kitchen.tracking.mlflow.search_experiments", side_effect=RuntimeError("network down")
    ):
        _verify_store_schema(f"sqlite:///{db}")  # must NOT raise — left for the command


def test_verify_store_schema_skips_remote_uri():
    with patch("kitchen.tracking.mlflow.search_experiments") as mock_search:
        _verify_store_schema("http://remote:5000")
    mock_search.assert_not_called()


def test_verify_store_schema_skips_missing_sqlite_file(tmp_path):
    with patch("kitchen.tracking.mlflow.search_experiments") as mock_search:
        _verify_store_schema(f"sqlite:///{tmp_path / 'does_not_exist.db'}")
    mock_search.assert_not_called()


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
    assert "tags.kitchen.data_sha256" not in runs.columns or runs.iloc[0].get(
        "tags.kitchen.data_sha256"
    ) != runs.iloc[0].get("tags.kitchen.data_sha256")  # NaN


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


# ── MNT-003: artifact-location drift ──────────────────────────────────────────


def _patch_version_source(monkeypatch, source, *, by_alias=True):
    """Make MlflowClient resolve a model version with the given .source."""
    client = MagicMock()
    mv = MagicMock(source=source)
    client.get_model_version_by_alias.return_value = mv
    client.get_model_version.return_value = mv
    monkeypatch.setattr("kitchen.tracking.mlflow.tracking.MlflowClient", lambda: client)
    return client


def test_local_path_from_uri_classifies_sources():
    assert _local_path_from_uri("file:///abs/path/model").as_posix() == "/abs/path/model"
    assert _local_path_from_uri("./mlruns/0/run/artifacts").as_posix().endswith("artifacts")
    assert _local_path_from_uri("s3://bucket/model") is None
    assert _local_path_from_uri("mlflow-artifacts:/0/run/artifacts/model") is None


def test_explain_drift_local_missing_source(monkeypatch, tmp_path):
    missing = tmp_path / "deleted"  # never created — guaranteed absent
    _patch_version_source(monkeypatch, f"file://{missing}")
    err = explain_model_load_error("models:/proj-model@champion", OSError("no such file"))
    assert isinstance(err, ArtifactLocationError)
    assert "not reachable from this environment" in str(err)
    assert str(missing) in str(err)
    assert "kitchen run train --auto-promote" in str(err)


def test_explain_no_drift_when_local_source_exists(monkeypatch, tmp_path):
    _patch_version_source(monkeypatch, f"file://{tmp_path}")
    assert explain_model_load_error("models:/proj-model@champion", OSError("boom")) is None


def test_explain_no_drift_for_s3_source(monkeypatch):
    _patch_version_source(monkeypatch, "s3://bucket/mlflow/0/abc/artifacts/model")
    # An unreachable S3 source is a different bug (perms/network) — not our case.
    assert explain_model_load_error("models:/proj-model@champion", OSError("denied")) is None


def test_explain_no_drift_for_s3_source_even_with_notfound_exc(monkeypatch):
    """A 'could not find'-style failure against an S3 source is perms/network, not drift.

    Pins the negative discriminator: the mlflow-artifacts missing-exc gate must not be
    widened to all non-local sources, or real S3 errors would false-positive as drift.
    """
    _patch_version_source(monkeypatch, "s3://bucket/mlflow/0/abc/artifacts/model")
    assert explain_model_load_error("models:/proj-model@champion", Exception("could not find")) is None


def test_explain_ignores_non_models_uri(monkeypatch):
    called = MagicMock()
    monkeypatch.setattr("kitchen.tracking.mlflow.tracking.MlflowClient", called)
    assert explain_model_load_error("runs:/abc123/model", OSError("x")) is None
    assert explain_model_load_error("s3://bucket/model", OSError("x")) is None
    called.assert_not_called()


def test_explain_mlflow_artifacts_source_gated_on_missing_exc(monkeypatch):
    _patch_version_source(monkeypatch, "mlflow-artifacts:/0/abc/artifacts/model")
    missing = explain_model_load_error(
        "models:/proj-model@champion", Exception("Failed to download artifacts")
    )
    assert isinstance(missing, ArtifactLocationError)
    # A benign-looking failure against a proxied source isn't claimed as drift.
    assert explain_model_load_error("models:/proj-model@champion", Exception("oom")) is None


def test_explain_version_uri_uses_get_model_version(monkeypatch):
    client = _patch_version_source(monkeypatch, "file:///gone/model")
    err = explain_model_load_error("models:/proj-model/3", OSError("no such file"))
    assert isinstance(err, ArtifactLocationError)
    client.get_model_version.assert_called_once_with("proj-model", "3")


def test_explain_follows_mlflow3_logged_model_indirection(monkeypatch, tmp_path):
    """MLflow 3.x: mv.source is a logged-model URI; the real path is its artifact_location."""
    missing = tmp_path / "deleted"  # never created — guaranteed absent
    client = MagicMock()
    client.get_model_version_by_alias.return_value = MagicMock(source="models:/m-abc123")
    client.get_logged_model.return_value = MagicMock(artifact_location=str(missing))
    monkeypatch.setattr("kitchen.tracking.mlflow.tracking.MlflowClient", lambda: client)

    err = explain_model_load_error("models:/proj-model@champion", OSError("no such file"))
    assert isinstance(err, ArtifactLocationError)
    assert str(missing) in str(err)
    client.get_logged_model.assert_called_once_with("m-abc123")


def test_explain_returns_none_when_lookup_fails(monkeypatch):
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = Exception("registry down")
    monkeypatch.setattr("kitchen.tracking.mlflow.tracking.MlflowClient", lambda: client)
    assert explain_model_load_error("models:/proj-model@champion", OSError("x")) is None
