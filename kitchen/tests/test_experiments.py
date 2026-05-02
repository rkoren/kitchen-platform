"""Tests for `kitchen experiments` and `kitchen promote` CLI commands."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()


def _make_run(
    run_id: str = "abcdef1234567890",
    name: str = "baseline",
    status: str = "FINISHED",
    start_time: int = 1_700_000_000_000,
    metrics: dict | None = None,
    tags: dict | None = None,
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = name
    run.info.status = status
    run.info.start_time = start_time
    run.data.metrics = metrics or {}
    run.data.tags = tags or {}
    return run


def _make_exp(experiment_id: str = "1") -> MagicMock:
    exp = MagicMock()
    exp.experiment_id = experiment_id
    return exp


# ---------------------------------------------------------------------------
# experiments list
# ---------------------------------------------------------------------------

class TestExperimentsList:
    def test_experiment_not_found(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            mock_client_cls.return_value.get_experiment_by_name.return_value = None
            result = runner.invoke(app, ["experiments", "list", "--experiment", "missing"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_no_runs(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = []
            result = runner.invoke(app, ["experiments", "list", "--experiment", "my-exp"])
        assert result.exit_code == 0
        assert "No runs" in result.output

    def test_shows_runs_with_metrics(self):
        runs = [
            _make_run("aaa0000011111111", "baseline", metrics={"val_accuracy": 0.85, "fi.feat1": 0.5}),
            _make_run("bbb0000022222222", "challenger", metrics={"val_accuracy": 0.88}),
        ]
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = runs
            result = runner.invoke(app, ["experiments", "list", "--experiment", "my-exp"])
        assert result.exit_code == 0
        assert "aaa00000" in result.output
        assert "bbb00000" in result.output
        assert "val_accuracy" in result.output
        # fi.* metrics should not appear as columns
        assert "fi." not in result.output

    def test_reads_experiment_from_params_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "params.yaml").write_text("experiment: from-yaml\n")
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = []
            result = runner.invoke(app, ["experiments", "list"])
        assert result.exit_code == 0
        assert "from-yaml" in result.output

    def test_fails_without_experiment_or_params(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["experiments", "list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# experiments compare
# ---------------------------------------------------------------------------

class TestExperimentsCompare:
    def test_experiment_not_found(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            mock_client_cls.return_value.get_experiment_by_name.return_value = None
            result = runner.invoke(app, ["experiments", "compare", "val_accuracy", "--experiment", "x"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_no_runs_with_metric(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = []
            result = runner.invoke(app, ["experiments", "compare", "val_accuracy", "--experiment", "x"])
        assert result.exit_code == 0
        assert "No runs" in result.output

    def test_ranks_runs_best_first(self):
        runs = [
            _make_run("aaa0000011111111", "winner", metrics={"val_accuracy": 0.92}),
            _make_run("bbb0000022222222", "loser", metrics={"val_accuracy": 0.80}),
        ]
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = runs
            result = runner.invoke(app, ["experiments", "compare", "val_accuracy", "--experiment", "x"])
        assert result.exit_code == 0
        assert "★" in result.output
        assert "aaa00000" in result.output
        # winner appears before loser
        assert result.output.index("aaa00000") < result.output.index("bbb00000")

    def test_shows_variant_tag(self):
        runs = [
            _make_run("aaa0000011111111", metrics={"val_accuracy": 0.9},
                      tags={"model_variant": "challenger"}),
        ]
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = runs
            result = runner.invoke(app, ["experiments", "compare", "val_accuracy", "--experiment", "x"])
        assert result.exit_code == 0
        assert "challenger" in result.output

    def test_lower_is_better_label(self):
        runs = [_make_run("aaa0000011111111", metrics={"val_brier": 0.1})]
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = runs
            result = runner.invoke(
                app,
                ["experiments", "compare", "val_brier", "--experiment", "x", "--lower-is-better"],
            )
        assert result.exit_code == 0
        assert "lower=better" in result.output


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------

class TestPromote:
    def _run_promote(self, *extra_args, run=None, production_uri=None):
        mock_run = run or _make_run(
            "abc1234567890000", "baseline",
            metrics={"val_accuracy": 0.9},
            tags={"model_variant": "baseline"},
        )
        with (
            patch("kitchen.tracking.configure_from_env"),
            patch("kitchen.registry.get_best_run", return_value=mock_run),
            patch("kitchen.registry.get_production_uri", return_value=production_uri),
            patch("kitchen.registry.register_model", return_value="1") as mock_reg,
            patch("kitchen.registry.promote_model") as mock_prom,
        ):
            result = runner.invoke(
                app,
                ["promote", "val_accuracy", "--experiment", "my-exp", *extra_args],
                catch_exceptions=False,
            )
            return result, mock_reg, mock_prom

    def test_dry_run_shows_winner(self):
        result, mock_reg, mock_prom = self._run_promote("--dry-run")
        assert result.exit_code == 0
        assert "abc12345" in result.output
        assert "Dry run" in result.output
        mock_reg.assert_not_called()
        mock_prom.assert_not_called()

    def test_registers_and_promotes(self):
        result, mock_reg, mock_prom = self._run_promote()
        assert result.exit_code == 0
        mock_reg.assert_called_once()
        mock_prom.assert_called_once()
        assert "Registered" in result.output
        assert "Promoted" in result.output
        assert "champion" in result.output

    def test_shows_current_champion_if_exists(self):
        result, _, _ = self._run_promote("--dry-run", production_uri="models:/my-exp-model@champion")
        assert "Current" in result.output
        assert "champion" in result.output

    def test_no_runs_exits_nonzero(self):
        with (
            patch("kitchen.tracking.configure_from_env"),
            patch("kitchen.registry.get_best_run", side_effect=ValueError("No runs found")),
        ):
            result = runner.invoke(app, ["promote", "val_accuracy", "--experiment", "x"])
        assert result.exit_code != 0
        assert "No runs found" in result.output

    def test_custom_model_name_and_alias(self):
        result, mock_reg, mock_prom = self._run_promote(
            "--model-name", "my-custom-model", "--alias", "staging"
        )
        assert result.exit_code == 0
        call_args = mock_reg.call_args
        assert call_args[0][2] == "my-custom-model"
        prom_call = mock_prom.call_args
        assert prom_call[1]["alias"] == "staging"

    def test_model_name_defaults_from_env(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_MODEL_NAME", "env-model-name")
        result, mock_reg, _ = self._run_promote()
        assert result.exit_code == 0
        assert mock_reg.call_args[0][2] == "env-model-name"
