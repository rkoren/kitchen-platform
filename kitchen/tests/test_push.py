"""Tests for `kitchen push` command (LML-005)."""
# pylint: disable=redefined-outer-name  # standard pytest fixture injection pattern

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import mlflow.exceptions
import pytest
from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def git_repo(tmp_path, monkeypatch):
    """Minimal git repo with a real commit so HEAD exists."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    # Need at least one commit so HEAD is valid
    (tmp_path / "README.md").write_text("hi\n")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    return tmp_path


def _git_sha(repo):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo
    ).decode().strip()


def _read_result_file(repo, branch, sha8):
    raw = subprocess.check_output(
        ["git", "show", f"{branch}:results/{sha8}.json"], cwd=repo
    ).decode()
    return json.loads(raw)


def _write_metrics(tmp_path, data):
    (tmp_path / "metrics.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_push_creates_results_branch(git_repo):
    _write_metrics(git_repo, {"val_accuracy": 0.91, "run_id": "abc123"})
    with patch("kitchen.tracking.configure_from_env"):
        result = runner.invoke(app, ["push"])
    assert result.exit_code == 0, result.output
    sha8 = _git_sha(git_repo)[:8]
    payload = _read_result_file(git_repo, "results", sha8)
    assert payload["sha"] == _git_sha(git_repo)
    assert payload["metrics"]["val_accuracy"] == pytest.approx(0.91)


def test_push_payload_schema(git_repo):
    _write_metrics(git_repo, {"val_accuracy": 0.88, "run_id": "run42"})
    with patch("kitchen.tracking.configure_from_env"):
        result = runner.invoke(app, ["push"])
    assert result.exit_code == 0
    sha8 = _git_sha(git_repo)[:8]
    payload = _read_result_file(git_repo, "results", sha8)
    assert set(payload.keys()) == {"sha", "timestamp", "run_id", "metrics", "lb_score", "champion"}
    assert payload["run_id"] == "run42"
    assert payload["lb_score"] is None
    assert payload["champion"] is False


def test_push_lb_score_extracted(git_repo):
    _write_metrics(git_repo, {"val_accuracy": 0.85, "kaggle_public_score": 0.77})
    with patch("kitchen.tracking.configure_from_env"):
        result = runner.invoke(app, ["push"])
    assert result.exit_code == 0
    sha8 = _git_sha(git_repo)[:8]
    payload = _read_result_file(git_repo, "results", sha8)
    assert payload["lb_score"] == pytest.approx(0.77)
    assert "kaggle_public_score" not in payload["metrics"]


def test_push_champion_flag(git_repo):
    run_id = "a" * 32
    _write_metrics(git_repo, {"val_accuracy": 0.91, "run_id": run_id})
    mv = MagicMock()
    mv.run_id = run_id
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient") as mock_cls,
    ):
        mock_cls.return_value.get_model_version_by_alias.return_value = mv
        result = runner.invoke(app, ["push", "--model-name", "my-model"])
    assert result.exit_code == 0
    sha8 = _git_sha(git_repo)[:8]
    payload = _read_result_file(git_repo, "results", sha8)
    assert payload["champion"] is True


def test_push_non_champion_flag(git_repo):
    run_id = "b" * 32
    champion_id = "c" * 32
    _write_metrics(git_repo, {"val_accuracy": 0.85, "run_id": run_id})
    mv = MagicMock()
    mv.run_id = champion_id
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient") as mock_cls,
    ):
        mock_cls.return_value.get_model_version_by_alias.return_value = mv
        result = runner.invoke(app, ["push", "--model-name", "my-model"])
    assert result.exit_code == 0
    sha8 = _git_sha(git_repo)[:8]
    payload = _read_result_file(git_repo, "results", sha8)
    assert payload["champion"] is False


def test_push_custom_branch(git_repo):
    _write_metrics(git_repo, {"val_accuracy": 0.80})
    with patch("kitchen.tracking.configure_from_env"):
        result = runner.invoke(app, ["push", "--branch", "my-branch"])
    assert result.exit_code == 0
    sha8 = _git_sha(git_repo)[:8]
    payload = _read_result_file(git_repo, "my-branch", sha8)
    assert "val_accuracy" in payload["metrics"]


def test_push_run_id_override(git_repo):
    _write_metrics(git_repo, {"val_accuracy": 0.80, "run_id": "original"})
    with patch("kitchen.tracking.configure_from_env"):
        result = runner.invoke(app, ["push", "--run-id", "override123"])
    assert result.exit_code == 0
    sha8 = _git_sha(git_repo)[:8]
    payload = _read_result_file(git_repo, "results", sha8)
    assert payload["run_id"] == "override123"


def test_push_idempotent_new_file_each_commit(git_repo):
    """Two successive pushes from the same repo produce two commits on the branch."""
    _write_metrics(git_repo, {"val_accuracy": 0.80})
    with patch("kitchen.tracking.configure_from_env"):
        runner.invoke(app, ["push"])

    # Make a new commit so HEAD changes
    (git_repo / "file2.txt").write_text("x\n")
    subprocess.run(["git", "add", "file2.txt"], cwd=git_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "step2"],
        cwd=git_repo, check=True, capture_output=True,
    )
    _write_metrics(git_repo, {"val_accuracy": 0.82})

    with patch("kitchen.tracking.configure_from_env"):
        runner.invoke(app, ["push"])

    log = subprocess.check_output(
        ["git", "log", "--oneline", "results"], cwd=git_repo
    ).decode()
    assert len(log.strip().splitlines()) == 2


def test_push_reads_experiment_from_params_yaml(git_repo):
    (git_repo / "params.yaml").write_text("experiment: my-exp\n")
    _write_metrics(git_repo, {"val_accuracy": 0.80})
    with patch("kitchen.tracking.configure_from_env"):
        result = runner.invoke(app, ["push"])
    assert result.exit_code == 0
    # Experiment name is embedded in the commit message on the results branch
    log = subprocess.check_output(
        ["git", "log", "--oneline", "results"], cwd=git_repo
    ).decode()
    assert "my-exp" in log


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_push_no_metrics_file(git_repo):  # pylint: disable=unused-argument
    with patch("kitchen.tracking.configure_from_env"):
        result = runner.invoke(app, ["push"])
    assert result.exit_code != 0
    assert "not found" in result.output or "not found" in (result.exception and str(result.exception) or "")


def test_push_registry_error_still_succeeds(git_repo):
    """Champion lookup failure is non-fatal."""
    _write_metrics(git_repo, {"val_accuracy": 0.80, "run_id": "xyz"})
    with (
        patch("kitchen.tracking.configure_from_env"),
        patch("mlflow.tracking.MlflowClient") as mock_cls,
    ):
        mock_cls.return_value.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException("not found")
        result = runner.invoke(app, ["push", "--model-name", "my-model"])
    assert result.exit_code == 0
    sha8 = _git_sha(git_repo)[:8]
    payload = _read_result_file(git_repo, "results", sha8)
    assert payload["champion"] is False
