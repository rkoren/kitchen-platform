"""Tests for `kitchen status` command (LML-002)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlflow.exceptions
from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()


def _make_run(
    run_id: str = "abcdef1234567890abcdef1234567890",
    metrics: dict | None = None,
    tags: dict | None = None,
    start_time: int = 1_700_000_000_000,
    run_name: str = "",
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = run_name
    run.info.start_time = start_time
    run.data.metrics = metrics or {}
    run.data.tags = tags or {}
    return run


def _make_mv(run_id: str, version: str = "3") -> MagicMock:
    mv = MagicMock()
    mv.run_id = run_id
    mv.version = version
    return mv


def _invoke(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    *extra_args: str,
    runs: list | None = None,
    champion_run_id: str | None = None,
    champion_metrics: dict | None = None,
    champion_tags: dict | None = None,
    exp_found: bool = True,
    registry_error: bool = False,
):
    if runs is None:
        runs = []

    champ_run = _make_run(
        champion_run_id or "cccccccccccccccccccccccccccccccc",
        metrics=champion_metrics or {"val_accuracy": 0.9},
        tags=champion_tags or {},
    )

    def _side_effect_alias(_model_name, _alias):
        if registry_error:
            raise mlflow.exceptions.MlflowException("not found")
        if champion_run_id is None:
            raise mlflow.exceptions.MlflowException("not found")
        return _make_mv(champion_run_id)

    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
    ):
        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = MagicMock(experiment_id="1") if exp_found else None
        client.search_runs.return_value = runs
        client.get_model_version_by_alias.side_effect = _side_effect_alias
        client.get_run.return_value = champ_run
        result = runner.invoke(app, ["status", "--experiment", "my-exp", *extra_args])
    return result


# ---------------------------------------------------------------------------
# Champion section
# ---------------------------------------------------------------------------


def test_no_champion_fallback():
    result = _invoke()
    assert result.exit_code == 0
    assert "no champion" in result.output.lower() or "no champion" in result.output


def test_champion_shown():
    champ_id = "a" * 32
    result = _invoke(champion_run_id=champ_id, champion_metrics={"val_accuracy": 0.91})
    assert result.exit_code == 0
    assert champ_id[:8] in result.output
    assert "champion" in result.output
    assert "val_accuracy" in result.output


def test_champion_version_shown():
    champ_id = "a" * 32
    result = _invoke(champion_run_id=champ_id)
    assert result.exit_code == 0
    assert "v3" in result.output


def test_champion_variant_shown():
    champ_id = "a" * 32
    result = _invoke(
        champion_run_id=champ_id,
        champion_tags={"model_variant": "challenger"},
    )
    assert result.exit_code == 0
    assert "challenger" in result.output


def test_champion_fi_metrics_excluded():
    champ_id = "a" * 32
    result = _invoke(
        champion_run_id=champ_id,
        champion_metrics={"val_accuracy": 0.9, "fi.feat1": 0.5},
    )
    assert result.exit_code == 0
    assert "fi." not in result.output


def test_registry_error_graceful():
    result = _invoke(registry_error=True)
    assert result.exit_code == 0
    assert "no champion" in result.output.lower() or "Run `kitchen promote" in result.output


# ---------------------------------------------------------------------------
# Recent runs section
# ---------------------------------------------------------------------------


def test_no_runs_message():
    result = _invoke()
    assert result.exit_code == 0
    assert "No runs found" in result.output


def test_runs_listed():
    runs = [
        _make_run("a" * 32, metrics={"val_accuracy": 0.85}),
        _make_run("b" * 32, metrics={"val_accuracy": 0.88}),
    ]
    result = _invoke(runs=runs)
    assert result.exit_code == 0
    assert "a" * 8 in result.output
    assert "b" * 8 in result.output


def test_champion_marked_in_table():
    champ_id = "a" * 32
    runs = [
        _make_run(champ_id, metrics={"val_accuracy": 0.85}),
        _make_run("b" * 32, metrics={"val_accuracy": 0.88}),
    ]
    result = _invoke(runs=runs, champion_run_id=champ_id)
    assert result.exit_code == 0
    # Only examine lines in the "Recent Runs" section (after the header)
    recent_section = result.output.split("Recent Runs")[-1]
    champ_line = next(
        line for line in recent_section.splitlines() if "a" * 8 in line
    )
    assert "[C]" in champ_line


def test_experiment_not_found_shows_message():
    result = _invoke(exp_found=False)
    assert result.exit_code == 0
    assert "not found" in result.output or "No experiment" in result.output


def test_model_name_defaults_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "custom-model")
    result = _invoke()
    assert result.exit_code == 0


def test_custom_n_runs():
    runs = [_make_run(f"{'a' * 30}{i:02d}", metrics={"val_accuracy": 0.8}) for i in range(3)]
    result = _invoke("--runs", "3", runs=runs)
    assert result.exit_code == 0
    assert "last 3" in result.output


# ---------------------------------------------------------------------------
# Threshold section
# ---------------------------------------------------------------------------


def test_threshold_pass_shown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(
        "experiment: my-exp\nthresholds:\n  val_accuracy: 0.80\n"
    )
    runs = [_make_run("a" * 32, metrics={"val_accuracy": 0.91})]
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
    ):
        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        client.search_runs.return_value = runs
        client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("nf")
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_threshold_fail_shown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(
        "experiment: my-exp\nthresholds:\n  val_accuracy: 0.95\n"
    )
    runs = [_make_run("a" * 32, metrics={"val_accuracy": 0.80})]
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
    ):
        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        client.search_runs.return_value = runs
        client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("nf")
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0  # status never exits non-zero
    assert "FAIL" in result.output


def test_threshold_fail_exits_zero(tmp_path, monkeypatch):
    """Status is informational; threshold failures must never make it exit non-zero."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text(
        "experiment: my-exp\nthresholds:\n  val_accuracy: 0.99\n"
    )
    runs = [_make_run("a" * 32, metrics={"val_accuracy": 0.50})]
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
    ):
        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        client.search_runs.return_value = runs
        client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("nf")
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0


def test_no_status_column_without_thresholds():
    runs = [_make_run("a" * 32, metrics={"val_accuracy": 0.85})]
    result = _invoke(runs=runs)
    assert result.exit_code == 0
    assert "STATUS" not in result.output
    assert "PASS" not in result.output
    assert "FAIL" not in result.output


# ---------------------------------------------------------------------------
# Submission file section
# ---------------------------------------------------------------------------


def test_submission_file_shown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sub_dir = tmp_path / "submissions"
    sub_dir.mkdir()
    (sub_dir / "submission.csv").write_text("id,target\n1,0\n")
    result = _invoke()
    assert result.exit_code == 0
    assert "submission.csv" in result.output


def test_no_submission_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _invoke()
    assert result.exit_code == 0
    assert "submission.csv" not in result.output


# ---------------------------------------------------------------------------
# Experiment resolution
# ---------------------------------------------------------------------------


def test_reads_experiment_from_params_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: from-yaml\n")
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
    ):
        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        client.search_runs.return_value = []
        client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("nf")
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "from-yaml" in result.output


def test_fails_without_experiment_or_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("kitchen.tracking.configure_from_env"):
        result = runner.invoke(app, ["status"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Data staleness section (LML-009)
# ---------------------------------------------------------------------------

import os  # noqa: E402

from kitchen.cli import _data_status  # noqa: E402


def _touch(path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    os.utime(path, (mtime, mtime))


def test_data_status_none_without_data_dir(tmp_path):
    assert _data_status(tmp_path) is None


def test_data_status_missing_when_processed_empty(tmp_path):
    _touch(tmp_path / "data" / "raw" / "train.csv", 1000)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    state, hint = _data_status(tmp_path)
    assert state == "missing"
    assert "kitchen run features" in hint


def test_data_status_fresh_when_processed_newer(tmp_path):
    _touch(tmp_path / "data" / "raw" / "train.csv", 1000)
    _touch(tmp_path / "data" / "processed" / "features.parquet", 2000)
    state, _ = _data_status(tmp_path)
    assert state == "fresh"


def test_data_status_stale_when_raw_newer(tmp_path):
    _touch(tmp_path / "data" / "processed" / "features.parquet", 1000)
    _touch(tmp_path / "data" / "raw" / "train.csv", 2000)
    state, hint = _data_status(tmp_path)
    assert state == "stale"
    assert "train.csv" in hint


def test_data_status_stale_when_feature_script_newer(tmp_path):
    _touch(tmp_path / "data" / "raw" / "train.csv", 1000)
    _touch(tmp_path / "data" / "processed" / "features.parquet", 1500)
    _touch(tmp_path / "src" / "features" / "run.py", 2000)
    state, hint = _data_status(tmp_path)
    assert state == "stale"
    assert "run.py" in hint


def test_data_status_dvc_hint_when_dvc_yaml_present(tmp_path):
    _touch(tmp_path / "data" / "processed" / "features.parquet", 1000)
    _touch(tmp_path / "data" / "raw" / "train.csv", 2000)
    (tmp_path / "dvc.yaml").write_text("stages: {}\n")
    _, hint = _data_status(tmp_path)
    assert "dvc repro" in hint


def test_status_renders_data_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: my-exp\n")
    _touch(tmp_path / "data" / "raw" / "train.csv", 2000)
    _touch(tmp_path / "data" / "processed" / "features.parquet", 1000)  # stale
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
    ):
        client = mock_client_cls.return_value
        client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        client.search_runs.return_value = []
        client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("nf")
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Data" in result.output
    assert "STALE" in result.output
