"""Tests for kitchen.experiment context manager (NB-001)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import kitchen
from kitchen.experiment import (
    ExperimentRun,
    _find_params_yaml,
    _seed_env_from_params_yaml,
    experiment,
    init_run,
)

# ── _find_params_yaml ─────────────────────────────────────────────────────────


def test_find_params_yaml_finds_in_cwd(tmp_path):
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    result = _find_params_yaml(tmp_path)
    assert result == tmp_path / "params.yaml"


def test_find_params_yaml_finds_in_parent(tmp_path):
    (tmp_path / "params.yaml").write_text("experiment: test\n")
    child = tmp_path / "subdir"
    child.mkdir()
    result = _find_params_yaml(child)
    assert result == tmp_path / "params.yaml"


def test_find_params_yaml_returns_none_when_absent(tmp_path):
    assert _find_params_yaml(tmp_path) is None


# ── _seed_env_from_params_yaml ────────────────────────────────────────────────


def test_seed_env_sets_tracking_uri(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    (tmp_path / "params.yaml").write_text("mlflow:\n  tracking_uri: http://localhost:5000\n")
    _seed_env_from_params_yaml(tmp_path / "params.yaml")
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://localhost:5000"


def test_seed_env_does_not_override_existing_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://existing:9999")
    (tmp_path / "params.yaml").write_text("mlflow:\n  tracking_uri: http://localhost:5000\n")
    _seed_env_from_params_yaml(tmp_path / "params.yaml")
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://existing:9999"


def test_seed_env_silently_ignores_bad_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    (tmp_path / "params.yaml").write_text("{{not valid yaml")
    _seed_env_from_params_yaml(tmp_path / "params.yaml")  # must not raise


# ── ExperimentRun ─────────────────────────────────────────────────────────────


def test_experiment_run_log_calls_log_metrics():
    active_run = MagicMock()
    active_run.info.run_id = "abc123"
    er = ExperimentRun(active_run)
    with patch("kitchen.experiment.mlflow") as mock_mlflow:
        er.log(accuracy=0.9, loss=0.1)
        mock_mlflow.log_metrics.assert_called_once_with({"accuracy": 0.9, "loss": 0.1})


def test_experiment_run_log_params_calls_log_params():
    active_run = MagicMock()
    er = ExperimentRun(active_run)
    with patch("kitchen.experiment.mlflow") as mock_mlflow:
        er.log_params(max_depth=6, eta=0.05)
        mock_mlflow.log_params.assert_called_once_with({"max_depth": 6, "eta": 0.05})


def test_experiment_run_set_tag():
    active_run = MagicMock()
    er = ExperimentRun(active_run)
    with patch("kitchen.experiment.mlflow") as mock_mlflow:
        er.set_tag("note", "baseline")
        mock_mlflow.set_tag.assert_called_once_with("note", "baseline")


def test_experiment_run_id():
    active_run = MagicMock()
    active_run.info.run_id = "run-xyz"
    er = ExperimentRun(active_run)
    assert er.run_id == "run-xyz"


# ── experiment() context manager ──────────────────────────────────────────────


def _make_mock_mlflow(run_id: str = "test-run-id"):
    """Return a mock mlflow module with a working start_run context manager."""
    mock = MagicMock()
    active_run = MagicMock()
    active_run.info.run_id = run_id
    mock.start_run.return_value.__enter__ = MagicMock(return_value=active_run)
    mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
    return mock, active_run


def test_experiment_yields_experiment_run():
    mock_mlflow, _ = _make_mock_mlflow()
    with (
        patch("kitchen.experiment.mlflow", mock_mlflow),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment.log_run_context"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
    ):
        with experiment("my-project") as run:
            assert isinstance(run, ExperimentRun)
            assert run.run_id == "test-run-id"


def test_experiment_starts_run_with_run_name():
    mock_mlflow, _ = _make_mock_mlflow()
    with (
        patch("kitchen.experiment.mlflow", mock_mlflow),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment.log_run_context"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
    ):
        with experiment("proj", run_name="trial-1"):
            mock_mlflow.start_run.assert_called_once_with(run_name="trial-1")


def test_experiment_logs_params_when_provided():
    mock_mlflow, _ = _make_mock_mlflow()
    with (
        patch("kitchen.experiment.mlflow", mock_mlflow),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment.log_run_context"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
    ):
        with experiment("proj", params={"model": {"max_depth": 6}}):
            mock_mlflow.log_params.assert_called_once_with({"model.max_depth": 6})


def test_experiment_calls_log_run_context():
    mock_mlflow, _ = _make_mock_mlflow()
    with (
        patch("kitchen.experiment.mlflow", mock_mlflow),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment.log_run_context") as mock_lrc,
        patch("kitchen.experiment._find_params_yaml", return_value=None),
    ):
        with experiment("proj"):
            mock_lrc.assert_called_once()


def test_experiment_falls_back_to_sqlite_on_configure_error():
    mock_mlflow, _ = _make_mock_mlflow()
    with (
        patch("kitchen.experiment.mlflow", mock_mlflow),
        patch("kitchen.experiment.configure_from_env", side_effect=Exception("unreachable")),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment.log_run_context"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
    ):
        with pytest.warns(UserWarning, match="falling back to sqlite"):
            with experiment("proj"):
                pass
        mock_mlflow.set_tracking_uri.assert_called_with("sqlite:///mlruns.db")


def test_experiment_seeds_from_params_yaml(tmp_path, monkeypatch):
    """params.yaml tracking_uri is applied when env var is absent."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ARTIFACT_BUCKET", raising=False)
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("mlflow:\n  tracking_uri: http://custom:5001\n")

    mock_mlflow, _ = _make_mock_mlflow()
    with (
        patch("kitchen.experiment.mlflow", mock_mlflow),
        patch("kitchen.experiment.configure_from_env") as mock_cfg,
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment.log_run_context"),
        patch("kitchen.experiment._find_params_yaml", return_value=params_yaml),
    ):
        with experiment("proj"):
            pass
        mock_cfg.assert_called_once()
        assert os.environ.get("MLFLOW_TRACKING_URI") == "http://custom:5001"


def test_experiment_env_var_wins_over_params_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-wins:9999")
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("mlflow:\n  tracking_uri: http://should-be-ignored:5001\n")

    mock_mlflow, _ = _make_mock_mlflow()
    with (
        patch("kitchen.experiment.mlflow", mock_mlflow),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment.log_run_context"),
        patch("kitchen.experiment._find_params_yaml", return_value=params_yaml),
    ):
        with experiment("proj"):
            pass
        assert os.environ["MLFLOW_TRACKING_URI"] == "http://env-wins:9999"


# ── Public API surface ────────────────────────────────────────────────────────


def test_kitchen_experiment_is_callable():
    """kitchen.experiment should be the function, not the module."""
    assert callable(kitchen.experiment)


# ── init_run() ────────────────────────────────────────────────────────────────


def _make_tracker_mock():
    """Return a mock Tracker whose .run() context manager yields an active_run."""
    tracker = MagicMock()
    active_run = MagicMock()
    active_run.info.run_id = "init-run-id"
    tracker.run.return_value.__enter__ = MagicMock(return_value=active_run)
    tracker.run.return_value.__exit__ = MagicMock(return_value=False)
    return tracker


def test_init_run_yields_tracker():
    mock_tracker = _make_tracker_mock()
    with (
        patch("kitchen.experiment.mlflow"),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
        patch("kitchen.experiment.Tracker", return_value=mock_tracker),
    ):
        with init_run({"experiment": "proj"}) as tracker:
            assert tracker is mock_tracker


def test_init_run_opens_tracker_run_with_params():
    mock_tracker = _make_tracker_mock()
    params = {"experiment": "proj", "model": {"depth": 4}}
    with (
        patch("kitchen.experiment.mlflow"),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
        patch("kitchen.experiment.Tracker", return_value=mock_tracker),
    ):
        with init_run(params, run_name="nb-trial"):
            mock_tracker.run.assert_called_once_with(run_name="nb-trial", params=params)


def test_init_run_uses_experiment_from_params():
    mock_tracker = _make_tracker_mock()
    with (
        patch("kitchen.experiment.mlflow"),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
        patch("kitchen.experiment.Tracker") as MockTracker,
    ):
        MockTracker.return_value = mock_tracker
        with init_run({"experiment": "my-cbb"}):
            MockTracker.assert_called_once_with("my-cbb")


def test_init_run_defaults_to_default_experiment_when_no_params():
    mock_tracker = _make_tracker_mock()
    with (
        patch("kitchen.experiment.mlflow"),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
        patch("kitchen.experiment.Tracker") as MockTracker,
    ):
        MockTracker.return_value = mock_tracker
        with init_run():
            MockTracker.assert_called_once_with("default")


def test_init_run_seeds_tracking_uri_from_params_dict(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    mock_tracker = _make_tracker_mock()
    params = {"experiment": "proj", "mlflow": {"tracking_uri": "http://from-dict:5000"}}
    with (
        patch("kitchen.experiment.mlflow"),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
        patch("kitchen.experiment.Tracker", return_value=mock_tracker),
    ):
        with init_run(params):
            pass
    assert os.environ.get("MLFLOW_TRACKING_URI") == "http://from-dict:5000"


def test_init_run_env_var_wins_over_params_dict(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-wins:9999")
    mock_tracker = _make_tracker_mock()
    params = {"experiment": "proj", "mlflow": {"tracking_uri": "http://ignored:5000"}}
    with (
        patch("kitchen.experiment.mlflow"),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
        patch("kitchen.experiment.Tracker", return_value=mock_tracker),
    ):
        with init_run(params):
            pass
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://env-wins:9999"


def test_init_run_auto_discovers_params_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("experiment: auto-proj\nmlflow:\n  tracking_uri: http://auto:5000\n")
    mock_tracker = _make_tracker_mock()
    with (
        patch("kitchen.experiment.mlflow"),
        patch("kitchen.experiment.configure_from_env"),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment._find_params_yaml", return_value=params_yaml),
        patch("kitchen.experiment.Tracker") as MockTracker,
    ):
        MockTracker.return_value = mock_tracker
        with init_run():
            MockTracker.assert_called_once_with("auto-proj")
    assert os.environ.get("MLFLOW_TRACKING_URI") == "http://auto:5000"


def test_init_run_falls_back_to_sqlite_on_configure_error():
    mock_mlflow = MagicMock()
    mock_tracker = _make_tracker_mock()
    with (
        patch("kitchen.experiment.mlflow", mock_mlflow),
        patch("kitchen.experiment.configure_from_env", side_effect=Exception("unreachable")),
        patch("kitchen.experiment.init_experiment"),
        patch("kitchen.experiment._find_params_yaml", return_value=None),
        patch("kitchen.experiment.Tracker", return_value=mock_tracker),
    ):
        with pytest.warns(UserWarning, match="falling back to sqlite"):
            with init_run({"experiment": "proj"}):
                pass
    mock_mlflow.set_tracking_uri.assert_called_with("sqlite:///mlruns.db")


def test_kitchen_init_run_is_callable():
    assert callable(kitchen.init_run)
