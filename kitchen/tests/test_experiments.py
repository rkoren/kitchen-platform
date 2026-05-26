"""Tests for `kitchen experiments` and `kitchen promote` CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
            _make_run(
                "aaa0000011111111", "baseline", metrics={"val_accuracy": 0.85, "fi.feat1": 0.5}
            ),
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
            result = runner.invoke(
                app, ["experiments", "compare", "val_accuracy", "--experiment", "x"]
            )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_no_runs_with_metric(self):
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = []
            result = runner.invoke(
                app, ["experiments", "compare", "val_accuracy", "--experiment", "x"]
            )
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
            result = runner.invoke(
                app, ["experiments", "compare", "val_accuracy", "--experiment", "x"]
            )
        assert result.exit_code == 0
        assert "★" in result.output
        assert "aaa00000" in result.output
        # winner appears before loser
        assert result.output.index("aaa00000") < result.output.index("bbb00000")

    def test_shows_variant_tag(self):
        runs = [
            _make_run(
                "aaa0000011111111",
                metrics={"val_accuracy": 0.9},
                tags={"model_variant": "challenger"},
            ),
        ]
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = runs
            result = runner.invoke(
                app, ["experiments", "compare", "val_accuracy", "--experiment", "x"]
            )
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
# leaderboard
# ---------------------------------------------------------------------------


class TestLeaderboard:
    def _invoke(self, runs, *extra_args, exp_found=True):
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp() if exp_found else None
            client.search_runs.return_value = runs
            return runner.invoke(app, ["leaderboard", "--experiment", "cbb-tournament", *extra_args])

    def test_experiment_not_found(self):
        result = self._invoke([], exp_found=False)
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_no_runs_with_metric(self):
        result = self._invoke([])
        assert result.exit_code == 0
        assert "No runs" in result.output

    def test_ranks_runs_best_first_lower_is_better(self):
        runs = [
            _make_run("a" * 32, "winner", metrics={"loto_brier": 0.164}),
            _make_run("b" * 32, "loser", metrics={"loto_brier": 0.172}),
        ]
        result = self._invoke(runs)
        assert result.exit_code == 0
        assert "★" in result.output
        assert result.output.index("a" * 32) < result.output.index("b" * 32)
        assert "lower=better" in result.output

    def test_shows_full_run_id(self):
        run_id = "abcdef1234567890abcdef1234567890"
        runs = [_make_run(run_id, metrics={"loto_brier": 0.16})]
        result = self._invoke(runs)
        assert result.exit_code == 0
        assert run_id in result.output

    def test_shows_lb_score_column(self):
        runs = [
            _make_run("a" * 32, metrics={"loto_brier": 0.164, "lb_score": 0.183}),
            _make_run("b" * 32, metrics={"loto_brier": 0.172}),
        ]
        result = self._invoke(runs)
        assert result.exit_code == 0
        assert "lb_score" in result.output
        assert "0.183" in result.output

    def test_shows_model_variant(self):
        runs = [
            _make_run("a" * 32, metrics={"loto_brier": 0.16}, tags={"model_variant": "challenger"}),
        ]
        result = self._invoke(runs)
        assert result.exit_code == 0
        assert "challenger" in result.output

    def test_higher_is_better_flag(self):
        runs = [_make_run("a" * 32, metrics={"val_auc": 0.92})]
        result = self._invoke(runs, "--metric", "val_auc", "--higher-is-better")
        assert result.exit_code == 0
        assert "higher=better" in result.output

    def test_custom_metric(self):
        runs = [_make_run("a" * 32, metrics={"brier_2026": 0.15})]
        result = self._invoke(runs, "--metric", "brier_2026")
        assert result.exit_code == 0
        assert "brier_2026" in result.output

    def test_reads_experiment_from_params_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "params.yaml").write_text("experiment: from-yaml\n")
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = []
            result = runner.invoke(app, ["leaderboard"])
        assert result.exit_code == 0
        assert "from-yaml" in result.output

    # --- champion marker (LML-003) ---

    def _invoke_with_champion(self, runs, champion_run_id, *extra_args):
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = runs
            mv = MagicMock()
            mv.run_id = champion_run_id
            client.get_model_version_by_alias.return_value = mv
            return runner.invoke(
                app, ["leaderboard", "--experiment", "cbb-tournament", *extra_args]
            )

    def test_champion_marked_when_not_rank1(self):
        """[C] marks the champion row even when it's not the top metric run."""
        champion_id = "b" * 32
        top_id = "a" * 32
        runs = [
            _make_run(top_id, metrics={"loto_brier": 0.164}),
            _make_run(champion_id, metrics={"loto_brier": 0.172}),
        ]
        result = self._invoke_with_champion(runs, champion_id)
        assert result.exit_code == 0
        top_line = next(line for line in result.output.splitlines() if top_id in line)
        champ_line = next(line for line in result.output.splitlines() if champion_id in line)
        assert "★" in top_line
        assert "[C]" in champ_line
        assert "[C]" not in top_line

    def test_champion_and_rank1_combined_marker(self):
        """★[C] appears when champion is also the metric-rank-1 run."""
        champion_id = "a" * 32
        runs = [
            _make_run(champion_id, metrics={"loto_brier": 0.164}),
            _make_run("b" * 32, metrics={"loto_brier": 0.172}),
        ]
        result = self._invoke_with_champion(runs, champion_id)
        assert result.exit_code == 0
        champ_line = next(line for line in result.output.splitlines() if champion_id in line)
        assert "★[C]" in champ_line

    def test_no_registered_model_graceful_fallback(self):
        """Registry lookup failure → normal leaderboard with ★, no crash, no [C]."""
        import mlflow.exceptions

        runs = [_make_run("a" * 32, metrics={"loto_brier": 0.164})]
        with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.get_experiment_by_name.return_value = _make_exp()
            client.search_runs.return_value = runs
            client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
                "not found"
            )
            result = runner.invoke(
                app, ["leaderboard", "--experiment", "cbb-tournament"]
            )
        assert result.exit_code == 0
        assert "★" in result.output
        assert "[C]" not in result.output

    def test_champion_footer_shows_model_name(self):
        """Footer references the model name passed via --model-name."""
        champion_id = "a" * 32
        runs = [_make_run(champion_id, metrics={"loto_brier": 0.164})]
        result = self._invoke_with_champion(
            runs, champion_id, "--model-name", "my-cbb-model"
        )
        assert result.exit_code == 0
        assert "my-cbb-model@champion" in result.output


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


class TestPromote:
    def _run_promote(self, *extra_args, run=None, production_uri=None):
        mock_run = run or _make_run(
            "abc1234567890000",
            "baseline",
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
        # K-016: confirm "Promoted" line contains the numeric version string, not a function repr
        promoted_line = next(l for l in result.output.splitlines() if "Promoted" in l)
        assert "v1" in promoted_line, f"Expected version string in: {promoted_line!r}"
        assert "<function" not in promoted_line, f"Bug: function object in: {promoted_line!r}"

    def test_shows_current_champion_if_exists(self):
        result, _, _ = self._run_promote(
            "--dry-run", production_uri="models:/my-exp-model@champion"
        )
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
